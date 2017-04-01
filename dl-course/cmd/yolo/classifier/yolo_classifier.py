# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import argparse
import math
import random
import numpy as np
import cv2
import json

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from yolo import *
from image_process import *

xp = np


def load_catalog(catalog_file):
    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return [], []

    dataset = catalog['dataset']
    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    truth_labels = np.asarray([[int(clazz) for clazz in item['classes']] for item in dataset])
    return image_paths, truth_labels

def one_epoch_train(model, optimizer, image_paths, truth_labels, batch_size):
    n_train = len(truth_labels)
    perm = np.random.permutation(n_train)

    loss, acc = (0., 0.)
    for count in six.moves.range(0, n_train, batch_size):
        ix = perm[count:count+batch_size]
        images = [Image(path, INPUT_SIZE, INPUT_SIZE).image for path in image_paths[ix]]
        labels = truth_labels[ix]

        xs = chainer.Variable(xp.asarray(images).astype(np.float32).transpose(0,3,1,2) / 255.)
        # TODO: マルチラベル画像対応 (いらないかも)
        ts = chainer.Variable(xp.asarray(labels.ravel()).astype(np.int32))

        model.train = True
        optimizer.update(model, xs, ts)
        print('mini-batch:%d loss:%f acc:%f' % ((count/batch_size)+1, model.loss.data, model.accuracy.data))
        loss += model.loss.data * len(ix) / n_train
        acc += model.accuracy.data * len(ix) / n_train
    return loss, acc

def one_epoch_cv(model, optimizer, image_paths, truth_labels):
    n_valid = len(truth_labels)

    loss, acc = (0., 0.)
    for count in six.moves.range(0, n_valid, 10):
        ix = np.arange(count, min(count+10, n_valid))
        images = [Image(path, INPUT_SIZE, INPUT_SIZE).image for path in image_paths[ix]]
        labels = truth_labels[ix]

        xs = chainer.Variable(xp.asarray(images).astype(np.float32).transpose(0,3,1,2) / 255.)
        # TODO: マルチラベル画像対応 (いらないかも)
        ts = chainer.Variable(xp.asarray(labels.ravel()).astype(np.int32))

        model.train = False
        model(xs, ts)
        loss += model.loss.data * len(ix) / n_valid
        acc += model.accuracy.data * len(ix) / n_valid
    return loss, acc

def train_model(args):
    print('train: gpu:%d epoch:%d batch_size:%d' %
        (args.gpu, args.n_epoch, args.batch_size))

    model = YoloClassifier(args.gpu)
    if len(args.model_file) > 0:
        print('load model: %s' % args.model_file)
        chainer.serializers.load_npz(args.model_file, model)

    # TODO: 本学習に合わせてMementum SGDとするか検討
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if len(args.state_file) > 0:
        print('load state: %s' % args.state_file)
        chainer.serializers.load_npz(args.state_file, optimizer)

    train_image_paths, train_labels = load_catalog(args.train_catalog_file)
    cv_image_paths, cv_labels = load_catalog(args.cv_catalog_file)
    print('number of dataset: train:%d cv:%d' % (len(train_labels), len(cv_labels)))

    logs = []
    for epoch in range(1, args.n_epoch+1):
        train_loss, train_acc = one_epoch_train(
            model, optimizer, train_image_paths, train_labels, args.batch_size)
        cv_loss, cv_acc = one_epoch_cv(
            model, optimizer, cv_image_paths, cv_labels)

        print('epoch:%d trian loss:%f train acc:%f cv loss:%f cv acc:%f' %
            (epoch, train_loss, train_acc, cv_loss, cv_acc))
        logs.append({
            'epoch': str(epoch),
            'train_loss': str(train_loss), 'train_acc': str(train_acc),
            'cv_loss': str(cv_loss), 'cv_acc': str(cv_acc)
        })
#        if (epoch % 10) == 0:
#            chainer.serializers.save_npz('classifier_epoch{}.model'.format(epoch), model)
#            chainer.serializers.save_npz('classifier_epoch{}.state'.format(epoch), optimizer)

    with open('classifier_train_log.json', 'w') as fp:
        json.dump({'epoch': str(args.n_epoch), 'batch_size': str(args.batch_size), 'logs': logs},
            fp, sort_keys=True, ensure_ascii=False, indent=2)
    chainer.serializers.save_npz('classifier_final.model', model)
    chainer.serializers.save_npz('classifier_final.state', optimizer)

def parse_arguments():
    description = 'YOLO Classifier'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-file', type=str, dest='model_file', default='')
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
