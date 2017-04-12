# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import argparse
import math
import random
import itertools
import datetime as dt
import numpy as np
import cv2
import json
import pandas as pd

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from yolo import *
from bounding_box import *
from image_process import *

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        dt.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
xp = np

# training configurations
learning_schedules = {
    '1'    : 1e-3,
    '500'  : 1e-3
}
momentum = 0.9
weight_decay = 0.0005


def load_catalog(catalog_file):
    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return []
    dataset = filter(lambda item: item['bounding_boxes'] != [], catalog['dataset'])
    return dataset

def dict_to_box(box):
    return Box(x=float(box['x']), y=float(box['y']),
            width=float(box['width']), height=float(box['height']),
            confidence=1., clazz=int(box['class']), objectness=1.)

def load_dataset(image_paths, truth_boxes):
    def load(path, boxes):
        image = Image(path, INPUT_SIZE, INPUT_SIZE)
        yolo_boxes = [real_to_yolo_coord(box, image.real_width, image.real_height)
                        for box in boxes]
        return {'image': image.image, 'truth': yolo_boxes}

    dataset = [load(path, boxes) for path, boxes in zip(image_paths, truth_boxes)]
    images = np.asarray([item['image'] for item in dataset])
    truth_boxes = [item['truth'] for item in dataset]
    return images, truth_boxes

def boxes_to_tensor(boxes):
    each_tensor = [encode_box_tensor(box) for box in boxes]
    return reduce(lambda x, y: x + y, each_tensor)

def tensor_to_boxes(tensor):
    return nms(select_candidates(tensor))

def init_positives():
    return [{'true': 0, 'false': 0}  for i in range(0, N_CLASSES)]

def count_positives(predicted_boxes, truth_boxes):
    positives = init_positives()
    for pred_box in predicted_boxes:
        if Box.correct(pred_box, truth_boxes):
            positives[int(pred_box.clazz)]['true'] += 1
        else:
            positives[int(pred_box.clazz)]['false'] += 1
    return positives

def add_positives(pos1, pos2):
    def add_item(item1, item2):
        return {'true': item1['true'] + item2['true'],
                'false': item1['false'] + item2['false']}
    return [add_item(item1, item2) for item1, item2 in zip(pos1, pos2)]

def average_precisions(positives):
    def precision(tp, fp):
        if tp == 0 and fp == 0:
            return 0.
        return float(tp) / (tp + fp)
    return [precision(p['true'], p['false']) for p in positives]

def mean_average_precision(positives):
    print('precision:{}'.format(positives))
    aps = average_precisions(positives)
    return np.asarray(aps).mean()

def perform_train(model, optimizer, dataset):
    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    real_truth_boxes = np.asarray([[dict_to_box(box) for box in item['bounding_boxes']]
                                    for item in dataset])
    images, truth_boxes = load_dataset(image_paths, real_truth_boxes)
    truth_tensors = np.asarray([boxes_to_tensor(boxes) for boxes in truth_boxes])

    xs = chainer.Variable(xp.asarray(images).transpose(0,3,1,2).astype(np.float32) / 255.)
    ts = chainer.Variable(xp.asarray(truth_tensors).astype(np.float32))

    model.train = True
    optimizer.update(model, xs, ts)
    return model.loss.data

def perform_cv(model, optimizer, dataset):
    n_valid = len(dataset)
    if n_valid == 0: return 0., 0.

    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    real_truth_boxes = np.asarray([[dict_to_box(box) for box in item['bounding_boxes']]
                                    for item in dataset])
    loss = 0.
    positives = init_positives()
    for count in six.moves.range(0, n_valid, 10):
        ix = np.arange(count, min(count+10, n_valid))
        images, truth_boxes = load_dataset(image_paths[ix], real_truth_boxes[ix])
        truth_tensors = np.asarray([boxes_to_tensor(boxes) for boxes in truth_boxes])

        xs = chainer.Variable(xp.asarray(images).transpose(0,3,1,2).astype(np.float32) / 255.)
        ts = chainer.Variable(xp.asarray(truth_tensors).astype(np.float32))

        model.train = False
        model(xs, ts)
        loss += model.loss.data * len(ix) / n_valid

        for batch in six.moves.range(0, len(ix)):
            predicted_boxes = tensor_to_boxes(model.h[batch])
            positives = add_positives(
                positives, count_positives(predicted_boxes, truth_boxes[batch]))
            for pred_box, truth_box in itertools.product(predicted_boxes, truth_boxes[batch]):
                print('{} {} pred:{} truth:{}'.format(
                    count + batch + 1, Box.correct(pred_box, [truth_box]), pred_box, truth_box))

    return loss, mean_average_precision(positives)

def train_model(args):
    print('train model: gpu:%d iteration:%d batch_size:%d' % \
        (args.gpu, args.iteration, args.batch_size))

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    model = YoloDetector(args.gpu)
    if len(args.model_file) > 0:
        print('load model: %s' % (args.model_file))
        chainer.serializers.load_npz(args.model_file, model)

    optimizer = chainer.optimizers.MomentumSGD(
        lr=learning_schedules['1'], momentum=momentum)
    optimizer.setup(model)
    if len(args.optimizer_file) > 0:
        print('load optimizer: %s' % (args.optimizer_file))
        chainer.serializers.load_npz(args.optimizer_file, optimizer)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    optimizer.use_cleargrads() # TODO: 必要？

    train_dataset = load_catalog(args.train_catalog_file)
    cv_dataset = load_catalog(args.cv_catalog_file)
    print('number of dataset: train:%d cv:%d' % (len(train_dataset), len(cv_dataset)))

    logs = []
    for iter_count in six.moves.range(1, args.iteration+1):
        if str(iter_count) in learning_schedules:
            optimizer.lr = learning_schedules[str(iter_count)]

        batch_dataset = np.random.choice(train_dataset, args.batch_size)
        train_loss = perform_train(model, optimizer, batch_dataset)
        print('mini-batch:%d %s' % (iter_count, model.loss_log))

        if (iter_count == 1) or (iter_count == args.iteration) or (iter_count % 100 == 0):
            cv_loss, cv_map = perform_cv(model, optimizer, cv_dataset)
            print('iter:%d trian loss:%f cv loss:%f map:%f' %
                (iter_count, train_loss, cv_loss, cv_map))
            logs.append({
                'iteration': str(iter_count),
                'train_loss': str(train_loss), 'train_map': str(0.0),
                'cv_loss': str(cv_loss), 'cv_map': str(cv_map)
            })
#            chainer.serializers.save_npz('detector_iter{}.model'.format(str(iter_count).zfill(5)), model)
#            chainer.serializers.save_npz('detector_iter{}.state'.format(str(iter_count).zfill(5)), optimizer)

            df_logs = pd.DataFrame(logs,
                columns=['iteration', 'train_loss', 'train_map', 'cv_loss', 'cv_map'])
            with open(os.path.join(SAVE_DIR, 'train_log.csv'), 'w') as fp:
                df_logs.to_csv(fp, encoding='cp932', index=False)

    if len(train_dataset) > 0:
        chainer.serializers.save_npz(os.path.join(SAVE_DIR, 'detector_final.model'), model)
        chainer.serializers.save_npz(os.path.join(SAVE_DIR, 'detector_final.state'), optimizer)

def parse_arguments():
    description = 'YOLO Detection Trainer'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-file', type=str, dest='model_file', default=DETECTOR_FIRST_MODEL_PATH)
    parser.add_argument('--optimizer-file', type=str, dest='optimizer_file', default='')
    parser.add_argument('--train-catalog-file', type=str, dest='train_catalog_file', default='')
    parser.add_argument('--cv-catalog-file', type=str, dest='cv_catalog_file', default='')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--iteration', '-i', type=int, dest='iteration', default=1)
    parser.add_argument('--batchsize', '-b', type=int, dest='batch_size', default=20)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy

    train_model(args)
    print('done')
