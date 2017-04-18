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
import pprint
import numpy as np
import cv2
import json
import pandas as pd

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from yolo_v2 import *
from bounding_box import *
from image_process import *
from collector import *

xp = np
pp = pprint.PrettyPrinter(indent=2)
pd.set_option('line_width', 150)

START_TIME = dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'snapshot_' + START_TIME)

def load_catalog(catalog_file):
    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return []
    dataset = filter(lambda item: item['bounding_boxes'] != [], catalog['dataset'])
    return dataset

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


def perform_train(model, optimizer, dataset):
    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    real_truth_boxes = np.asarray([[dict_to_box(box) for box in item['bounding_boxes']]
                                    for item in dataset])
    images, truth_boxes = load_dataset(image_paths, real_truth_boxes)

    xs = chainer.Variable(xp.asarray(images).transpose(0,3,1,2).astype(np.float32) / 255.)
    ts = [[yolo_to_grid_coord(box) for box in boxes] for boxes in truth_boxes]

    model.train = True
    optimizer.update(model, xs, ts)
    return model.loss.data

def perform_cv(model, optimizer, dataset):
    n_valid = len(dataset)
    if n_valid == 0: return 0., 0.
    collector = Collector(args.cv_catalog_file)

    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    real_truth_boxes = np.asarray([[dict_to_box(box) for box in item['bounding_boxes']]
                                    for item in dataset])
    loss = 0.
    for count in six.moves.range(0, n_valid, 10):
        ix = np.arange(count, min(count+10, n_valid))
        images, truth_boxes = load_dataset(image_paths[ix], real_truth_boxes[ix])

        xs = chainer.Variable(xp.asarray(images).transpose(0,3,1,2).astype(np.float32) / 255.)
        ts = [[yolo_to_grid_coord(box) for box in boxes] for boxes in truth_boxes]

        model.train = False
        model(xs, ts)
        loss += model.loss.data * len(ix) / n_valid
        tensors = model.h

        for batch in six.moves.range(0, len(ix)):
            bounding_boxes = inference_to_bounding_boxes(tensors[batch])
            candidates = select_candidates(bounding_boxes, CLASS_PROBABILITY_THRESH)
            winners = nms(candidates, NMS_IOU_THRESH)
            collector.validate_bounding_boxes(winners, truth_boxes[batch])
            for winner, truth_box in itertools.product(winners, truth_boxes[batch]):
                correct, iou = Box.correct(winner, [truth_box])
                print('{0} {1} {2:.3f} pred:{3} truth:{4}'.format(
                    count + batch + 1, correct, iou, winner, truth_box))

    collector.update()
    print(collector.df)
    print('map:%f recall:%f' % (collector.mean_ap, collector.recall))
    return loss, collector.mean_ap, collector.recall


def save_learning_params(args):
    params = {
        'elapsed_time': {
            'start': START_TIME,
            'end': dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        },
        'catalog_file': {
            'train': args.train_catalog_file,
            'cv': args.cv_catalog_file
        },
        'grid_cells': N_GRID,
        'anchor_boxes': '{}'.format(ANCHOR_BOXES),
        'max_iterations': args.iteration,
        'batch_size': args.batch_size,
        'momentum': MOMENTUM,
        'weight_decay': WEIGHT_DECAY,
        'lr_schedules': LR_SCHEDULES,
        'dropout_ratio': DROPOUT_RATIO,
        'scale_factors': SCALE_FACTORS,
        'confidence_keep_thresh': CONFIDENCE_KEEP_THRESH,
        'class_prob_thresh': CLASS_PROBABILITY_THRESH,
        'nms_iou_thresh': NMS_IOU_THRESH
    }
    with open(os.path.join(SAVE_DIR, 'params.json'), 'w') as fp:
        json.dump(params, fp, sort_keys=True, ensure_ascii=False, indent=2)


def train_model(args):
    print('train: gpu:%d iteration:%d batch_size:%d save_dir:%s' % \
          (args.gpu, args.iteration, args.batch_size, os.path.split(SAVE_DIR)[1]))

    model = YoloDetector(args.gpu)
    if len(args.model_file) > 0:
        print('load model: %s' % (args.model_file))
        chainer.serializers.load_npz(args.model_file, model)

    optimizer = chainer.optimizers.MomentumSGD(lr=LR_SCHEDULES['1'], momentum=MOMENTUM)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
    optimizer.use_cleargrads()
    if len(args.optimizer_file) > 0:
        print('load optimizer: %s' % (args.optimizer_file))
        chainer.serializers.load_npz(args.optimizer_file, optimizer)

    train_dataset = load_catalog(args.train_catalog_file)
    cv_dataset = load_catalog(args.cv_catalog_file)
    print('number of dataset: train:%d cv:%d' % (len(train_dataset), len(cv_dataset)))

    logs = []
    for iter_count in six.moves.range(args.start_iter_count,
                                      args.start_iter_count+args.iteration):
        if str(iter_count) in LR_SCHEDULES:
            optimizer.lr = LR_SCHEDULES[str(iter_count)]

        batch_dataset = np.random.choice(train_dataset, args.batch_size)
        model.iter_count = iter_count

        train_loss = perform_train(model, optimizer, batch_dataset)
        print('mini-batch:%d %s' % (iter_count, model.loss_log))

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            save_learning_params(args)
    
        if (iter_count == 10) or (iter_count % 100 == 0) or (iter_count == args.iteration):
            cv_loss, cv_map, cv_recall = perform_cv(model, optimizer, cv_dataset)
            print('iter:%d trian loss:%f cv loss:%f map:%f recall:%f' %
                  (iter_count, train_loss, cv_loss, cv_map, cv_recall))
            logs.append({
                'iteration': str(iter_count),
                'train_loss': str(train_loss), 'cv_loss': str(cv_loss),
                'cv_map': str(cv_map), 'cv_recall': str(cv_recall)
            })

            df_logs = pd.DataFrame(logs,
                        columns=['iteration', 'train_loss', 'cv_loss', 'cv_map', 'cv_recall'])
            with open(os.path.join(SAVE_DIR, 'train_log.csv'), 'w') as fp:
                df_logs.to_csv(fp, encoding='cp932', index=False)

        if iter_count % 1000 == 0:
            chainer.serializers.save_npz(
                os.path.join(SAVE_DIR, 'detector_iter{}.model'.format(str(iter_count).zfill(5))), model)
            chainer.serializers.save_npz(
                os.path.join(SAVE_DIR, 'detector_iter{}.state'.format(str(iter_count).zfill(5))), optimizer)

    if len(train_dataset) > 0:
        chainer.serializers.save_npz(os.path.join(SAVE_DIR, 'detector_final.model'), model)
        chainer.serializers.save_npz(os.path.join(SAVE_DIR, 'detector_final.state'), optimizer)

    save_learning_params(args)


def parse_arguments():
    description = 'YOLO Detection Trainer'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-file', type=str, dest='model_file', default=DETECTOR_FIRST_MODEL_PATH)
    parser.add_argument('--optimizer-file', type=str, dest='optimizer_file', default='')
    parser.add_argument('--train-catalog-file', type=str, dest='train_catalog_file', default='')
    parser.add_argument('--cv-catalog-file', type=str, dest='cv_catalog_file', default='')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--batchsize', '-b', type=int, dest='batch_size', default=20)
    parser.add_argument('--iteration', '-i', type=int, dest='iteration', default=1)
    parser.add_argument('--start-iteration-count', '-s', type=int, dest='start_iter_count', default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy

    train_model(args)
    print('done')
