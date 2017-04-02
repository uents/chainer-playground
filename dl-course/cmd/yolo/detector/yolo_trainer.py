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
import numpy as np
import cv2
import json

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from yolo import *
from bounding_box import *
from image_process import *

xp = np

# training configurations
learning_schedules = {
    '0'    : 1e-5,
    '500'  : 1e-4,
    '10000': 1e-5,
    '20000': 1e-6
}
momentum = 0.9
weight_decay = 0.005


def load_catalog(catalog_file):
    def dict_to_box(box):
        return Box(x=float(box['x']), y=float(box['y']),
                width=float(box['width']), height=float(box['height']),
                clazz=int(box['class']), objectness=1.)

    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return [], []

    dataset = filter(lambda item: item['bounding_boxes'] != [], catalog['dataset'])
    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    truth_boxes = np.asarray([[dict_to_box(box) for box in item['bounding_boxes']]
                                for item in dataset])
    return image_paths, truth_boxes

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

def one_epoch_train(model, optimizer, image_paths, ground_truth_boxes, batch_size, epoch):
    n_train = len(ground_truth_boxes)
    perm = np.random.permutation(n_train)

    loss = 0.
    positives = init_positives()
    for count in six.moves.range(0, n_train, batch_size):
#        ix = np.arange(count, min(count+batch_size, n_train))
        ix = perm[count:count+batch_size]
        images, truth_boxes = load_dataset(image_paths[ix], ground_truth_boxes[ix])
        truth_tensors = np.asarray([boxes_to_tensor(boxes) for boxes in truth_boxes])

        xs = chainer.Variable(xp.asarray(images).transpose(0,3,1,2).astype(np.float32) / 255.)
        ts = chainer.Variable(xp.asarray(truth_tensors).astype(np.float32))

        # TODO: Epoch単位のスケジュールに変更
        iteration = (epoch - 1) * batch_size + count
        if str(iteration) in learning_schedules:
            optimizer.lr = learning_schedules[str(iteration)]

        model.train = True
        optimizer.update(model, xs, ts)
        loss += model.loss.data * len(ix) / n_train

        for batch in six.moves.range(0, len(ix)):
            predicted_boxes = tensor_to_boxes(model.h[batch])
            positives = add_positives(
                positives, count_positives(predicted_boxes, truth_boxes[batch]))
#            for pred_box, truth_box in itertools.product(predicted_boxes, truth_boxes[batch]):
#                print('{} {} pred:{} truth:{}'.format(
#                    count + batch + 1, Box.correct(pred_box, truth_box), pred_box, truth_box))

    return loss, mean_average_precision(positives)

def one_epoch_cv(model, optimizer, image_paths, ground_truth_boxes):
    n_valid = len(ground_truth_boxes)

    loss = 0.
    positives = init_positives()
    for count in six.moves.range(0, n_valid, 10):
        ix = np.arange(count, min(count+10, n_valid))
        images, truth_boxes = load_dataset(image_paths[ix], ground_truth_boxes[ix])
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
    print('train model: gpu:%d epoch:%d batch_size:%d' % \
        (args.gpu, args.n_epoch, args.batch_size))

    model = YoloDetector(args.gpu)
    if len(args.model_file) > 0:
        print('load model: %s' % (args.model_file))
        chainer.serializers.load_npz(args.model_file, model)

    optimizer = chainer.optimizers.MomentumSGD(
        lr=learning_schedules['0'], momentum=momentum)
    optimizer.setup(model)
    if len(args.state_file) > 0:
        print('load state: %s' % (args.state_file))
        chainer.serializers.load_npz(args.state_file, optimizer)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    optimizer.use_cleargrads() # TODO: 必要？

    train_image_paths, train_boxes = load_catalog(args.train_catalog_file)
    cv_image_paths, cv_boxes = load_catalog(args.cv_catalog_file)
    print('number of dataset: train:%d cv:%d' % (len(train_boxes), len(cv_boxes)))

    logs = []
    for epoch in six.moves.range(1, args.n_epoch+1):
        train_loss, train_map = one_epoch_train(
            model, optimizer, train_image_paths, train_boxes, args.batch_size, epoch)
        cv_loss, cv_map = one_epoch_cv(
            model, optimizer, cv_image_paths, cv_boxes)

        print('epoch:%d trian loss:%f train map:%f cv loss:%f cv map:%f' %
            (epoch, train_loss, train_map, cv_loss, cv_map))
        logs.append({
            'epoch': str(epoch),
            'train_loss': str(train_loss), 'train_map': str(train_map),
            'cv_loss': str(cv_loss), 'cv_map': str(cv_map)
        })
#        if (epoch % 10) == 0:
#            chainer.serializers.save_npz('detector_epoch{}.model'.format(epoch), model)
#            chainer.serializers.save_npz('detector_epoch{}.state'.format(epoch), optimizer)

    with open('detector_train_log.json', 'w') as fp:
        json.dump({'epoch': str(args.n_epoch), 'batch_size': str(args.batch_size), 'logs': logs},
            fp, sort_keys=True, ensure_ascii=False, indent=2)
    chainer.serializers.save_npz('detector_final.model', model)
    chainer.serializers.save_npz('detector_final.state', optimizer)


def parse_arguments():
    description = 'YOLO Detection Trainer'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-file', type=str, dest='model_file', default=DETECTOR_FIRST_MODEL_PATH)
    parser.add_argument('--state-file', type=str, dest='state_file', default='')
    parser.add_argument('--train-catalog-file', type=str, dest='train_catalog_file', default='')
    parser.add_argument('--cv-catalog-file', type=str, dest='cv_catalog_file', default='')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--epoch', '-e', type=int, dest='n_epoch', default=1)
    parser.add_argument('--batchsize', '-b', type=int, dest='batch_size', default=20)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy

    train_model(args)
    print('done')
