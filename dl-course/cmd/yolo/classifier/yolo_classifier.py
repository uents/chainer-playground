# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import argparse
import math
import random
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
from yolo_v2 import *
from image import *

xp = np

START_TIME = dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'snapshot_' + START_TIME)

learning_rate = 0.01
lr_decay_power = 2
momentum = 0.9
weight_decay = 0.0005

def load_catalog(catalog_file):
    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return []
    return catalog['dataset']

def perform_train(model, optimizer, dataset):
    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    images = [Image(path, INPUT_SIZE).image for path in image_paths]
    # TODO: マルチラベル画像対応 (いらないかも)
    truth_labels = np.asarray([[int(clazz) for clazz in item['classes']] for item in dataset])

    xs = chainer.Variable(xp.asarray(images).astype(np.float32).transpose(0,3,1,2) / 255.)
    ts = chainer.Variable(xp.asarray(truth_labels.ravel()).astype(np.int32))

    model.train = True
    optimizer.update(model, xs, ts)
    return model.loss.data, model.accuracy.data

def perform_cv(model, optimizer, dataset):
    n_valid = len(dataset)
    if n_valid == 0: return 0., 0.

    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    # TODO: マルチラベル画像対応 (いらないかも)
    truth_labels = np.asarray([[int(clazz) for clazz in item['classes']] for item in dataset])

    loss, acc = (0., 0.)
    for count in six.moves.range(0, n_valid, 10):
        ix = np.arange(count, min(count+10, n_valid))
        images = [Image(path, INPUT_SIZE).image for path in image_paths[ix]]

        xs = chainer.Variable(xp.asarray(images).astype(np.float32).transpose(0,3,1,2) / 255.)
        ts = chainer.Variable(xp.asarray(truth_labels[ix].ravel()).astype(np.int32))

        model.train = False
        model(xs, ts)
        loss += model.loss.data * len(ix) / n_valid
        acc += model.accuracy.data * len(ix) / n_valid
    return loss, acc

def train_model(args):
    print('train: gpu:%d iteration:%d batch_size:%d save_dir:%s' % \
          (args.gpu, args.iteration, args.batch_size, os.path.split(SAVE_DIR)[1]))

    model = YoloClassifier(args.gpu)
    if len(args.model_file) > 0:
        print('load model: %s' % args.model_file)
        chainer.serializers.load_npz(args.model_file, model)

    optimizer = chainer.optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    optimizer.use_cleargrads()
    if len(args.optimizer_file) > 0:
        print('load optimizer: %s' % args.optimizer_file)
        chainer.serializers.load_npz(args.optimizer_file, optimizer)

    train_dataset = load_catalog(args.train_catalog_file)
    cv_dataset = load_catalog(args.cv_catalog_file)
    print('number of dataset: train:%d cv:%d' % (len(train_dataset), len(cv_dataset)))

    logs = []
    for iter_count in six.moves.range(1, args.iteration+1):
        batch_dataset = np.random.choice(train_dataset, args.batch_size)
        train_loss, train_acc = perform_train(model, optimizer, batch_dataset)
        print('mini-batch:%d loss:%f acc:%f' % (iter_count, train_loss, train_acc))

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
    
        if (iter_count == 1) or (iter_count == args.iteration) or (iter_count % 100 == 0):
            cv_loss, cv_acc = perform_cv(model, optimizer, cv_dataset)
            print('iter:%d train loss:%f acc:%f cv loss:%f acc:%f' %
                (iter_count, train_loss, train_acc, cv_loss, cv_acc))
            logs.append({
                'iteration': str(iter_count),
                'train_loss': str(train_loss), 'train_acc': str(train_acc),
                'cv_loss': str(cv_loss), 'cv_acc': str(cv_acc)
            })

            df_logs = pd.DataFrame(logs,
                columns=['iteration', 'train_loss', 'train_acc', 'cv_loss', 'cv_acc'])
            with open(os.path.join(SAVE_DIR, 'train_log.csv'), 'w') as fp:
                df_logs.to_csv(fp, encoding='cp932', index=False)

        if iter_count % 500 == 0:
            chainer.serializers.save_npz(
                os.path.join(SAVE_DIR, 'classifier_iter{}.model'.format(str(iter_count).zfill(5))), model)
            chainer.serializers.save_npz(
                os.path.join(SAVE_DIR, 'classifier_iter{}.state'.format(str(iter_count).zfill(5))), optimizer)

        # polynomial decay learning rate
        optimizer.lr = learning_rate * ((1-iter_count/5000.0) ** lr_decay_power)

    if len(train_dataset) > 0:
        chainer.serializers.save_npz(os.path.join(SAVE_DIR, 'classifier_final.model'), model)
        chainer.serializers.save_npz(os.path.join(SAVE_DIR, 'classifier_final.state'), optimizer)


def parse_arguments():
    description = 'YOLO Classifier'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-file', type=str, dest='model_file', default='')
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
