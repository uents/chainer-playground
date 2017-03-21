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
import scipy.io
import cv2
import json

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Function, Link

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from yolo import *
from image_process import *

# configurations
TRAIN_DATASET_PATH = os.path.join('.', 'train_dataset.npz')


def prepare_dataset(args):
    print('prepare dataset: catalog:%s' % (args.catalog_file))
    with open(os.path.join(args.catalog_file), 'r') as fp:
        catalog = json.load(fp)
    train_dataset = catalog['dataset']
    images = np.asarray([load_image(item['color_image_path'], INPUT_SIZE, INPUT_SIZE) for item in train_dataset])
    labels = np.asarray([item['classes'] for item in train_dataset]).astype(np.int32)
    print('save train dataset: images={}, ground_truths={}'.format(images.shape, ground_truths.shape))
    np.savez(TRAIN_DATASET_PATH, images=images, labels=labels)

def one_epoch_train(model, optimizer, images, labels, batch_size):
    n_train = len(labels)
    perm = np.random.permutation(n_train)

    sum_loss, sum_acc = (0., 0.)
    for count in six.moves.range(0, n_train, batch_size):
        ix = perm[count:count+batch_size]
        xs = chainer.Variable(xp.asarray(images[ix]).astype(np.float32).transpose(0,3,1,2))
        ts = chainer.Variable(xp.asarray(labels[ix].ravel()).astype(np.int32))

        model.train = True
        optimizer.update(model, xs, ts)
        print('mini-batch:%d loss:%f acc:%f' % ((count/batch_size)+1, model.loss.data, model.accuracy.data))
        sum_loss += model.loss.data * len(ix) / n_train
        sum_acc += model.accuracy.data * len(ix) / n_train
    return sum_loss, sum_acc

def one_epoch_cv(model, optimizer, images, labels):
    n_valid = len(labels)

    sum_loss, sum_acc = (0., 0.)
    for count in six.moves.range(0, n_valid, 10):
        ix = np.arange(count, count+10)
        xs = chainer.Variable(xp.asarray(images[ix]).astype(np.float32).transpose(0,3,1,2))
        ts = chainer.Variable(xp.asarray(labels[ix].ravel()).astype(np.int32))

        model.train = False
        model(xs, ts)
        sum_loss += model.loss.data * len(ix) / n_valid
        sum_acc += model.accuracy.data * len(ix) / n_valid
    return sum_loss, sum_acc

def train_model(args):
    print('train model: gpu:%d epoch:%d batch_size:%d init_model:%s init_state:%s' % \
        (args.gpu, args.n_epoch, args.batch_size, args.init_model_file, args.init_state_file))

    model = YoloTinyCNN()
    if len(args.init_model_file) > 0:
        chainer.serializers.load_npz(args.init_model_file, model)
    if args.gpu >= 0: model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if len(args.init_state_file) > 0:
        chainer.serializers.load_npz(args.init_state_file, optimizer)

    logs = []
    train_dataset = np.load(TRAIN_DATASET_PATH)
    images = train_dataset['images']
    labels = train_dataset['labels']

    train_ixs = sorted(random.sample(range(labels.shape[0]), int(labels.shape[0] * 0.8)))
    cv_ixs = sorted(list(set(range(labels.shape[0])) - set(train_ixs)))
    print('number of dataset: train:%d cv:%d' % (len(train_ixs), len(cv_ixs)))

    for epoch in range(1, args.n_epoch+1):
        train_loss, train_acc = one_epoch_train(model, optimizer, images[train_ixs], labels[train_ixs], args.batch_size)
        cv_loss, cv_acc = one_epoch_cv(model, optimizer, images[cv_ixs], labels[cv_ixs])
#        cv_loss, cv_acc = (0., 0.)
        print('epoch:%d trian loss:%f train acc:%f cv loss:%f cv acc:%f' % (epoch, train_loss, train_acc, cv_loss, cv_acc))
        logs.append({'epoch': str(epoch),
            'train_loss': str(train_loss), 'train_acc': str(train_acc),
            'cv_loss': str(cv_loss), 'cv_acc': str(cv_acc)})
        if (epoch % 10) == 0:
            chainer.serializers.save_npz('cnn_epoch{}.model'.format(epoch), model)
            chainer.serializers.save_npz('cnn_epoch{}.state'.format(epoch), optimizer)

    chainer.serializers.save_npz('cnn_final.model', model)
    chainer.serializers.save_npz('cnn_final.state', optimizer)
    with open('cnn_train_log.json', 'w') as fp:
        json.dump({'epoch': str(args.n_epoch), 'batch_size': str(args.batch_size), 'logs': logs},
            fp, sort_keys=True, ensure_ascii=False, indent=2)

def parse_arguments():
    usage = 'make training dataset catalog (sample code)'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--action', '-a', type=str, dest='action', required=True)
    # options for preparing dataset
    parser.add_argument('--catalog-file', type=str, dest='catalog_file')
    # options for training model
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--batchsize', '-b', type=int, dest='batch_size', default=20)
    parser.add_argument('--epoch', '-e', type=int, dest='n_epoch', default=1)
    parser.add_argument('--init-model-file', type=str, dest='init_model_file', default='')
    parser.add_argument('--init-state-file', type=str, dest='init_state_file', default='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy

    if args.action == 'prepare':
        prepare_dataset(args)
    elif args.action == 'train':
        train_model(args)
    print('done')
