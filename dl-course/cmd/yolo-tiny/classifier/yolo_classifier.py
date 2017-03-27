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
from yolo import *
from image_process import *

# configurations
DATASET_PATH = os.path.join('.', 'classifier_dataset.npz')


def parse_item_of_dataset(item):
    image = Image(item['color_image_path'], INPUT_SIZE, INPUT_SIZE).image
    label = [int(clazz) for clazz in item['classes']]
    return {'image': image, 'label': label}

def load_dataset(catalog_file):
    with open(os.path.join(catalog_file), 'r') as fp:
        catalog = json.load(fp)
    items = [parse_item_of_dataset(item) for item in catalog['dataset']]
    images = np.asarray([item['image'] for item in items])
    labels = np.asarray([item['label'] for item in items])
    return images, labels

def prepare_dataset(args):
    print('prepare dataset: train:%s cv:%s' %
        (args.train_catalog_file, args.cv_catalog_file))
    train_images, train_labels = load_dataset(args.train_catalog_file)
    cv_images, cv_labels = load_dataset(args.cv_catalog_file)
    print('save dataset: train images={} labels={}, cv images={} labels={}'
        .format(train_images.shape, train_labels.shape, cv_images.shape, cv_labels.shape))
    np.savez(DATASET_PATH, train_images=train_images, train_labels=train_labels,
        cv_images=cv_images, cv_labels=cv_labels)

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

    model = YoloClassifier(args.gpu)
    if len(args.init_model_file) > 0:
        chainer.serializers.load_npz(args.init_model_file, model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if len(args.init_state_file) > 0:
        chainer.serializers.load_npz(args.init_state_file, optimizer)

    logs = []
    dataset = np.load(DATASET_PATH)
    train_images = dataset['train_images']
    train_labels = dataset['train_labels']
    cv_images = dataset['cv_images']
    cv_labels = dataset['cv_labels']
    print('number of dataset: train:%d cv:%d' % (len(train_labels), len(cv_labels)))

    for epoch in range(1, args.n_epoch+1):
        train_loss, train_acc = one_epoch_train(
            model, optimizer, train_images, train_labels, args.batch_size)
        cv_loss, cv_acc = one_epoch_cv(
            model, optimizer, cv_images, cv_labels)
        print('epoch:%d trian loss:%f train acc:%f cv loss:%f cv acc:%f' %
            (epoch, train_loss, train_acc, cv_loss, cv_acc))
        logs.append({
            'epoch': str(epoch),
            'train_loss': str(train_loss), 'train_acc': str(train_acc),
            'cv_loss': str(cv_loss), 'cv_acc': str(cv_acc)
        })
        if (epoch % 10) == 0:
            chainer.serializers.save_npz('classifier_epoch{}.model'.format(epoch), model)
            chainer.serializers.save_npz('classifier_epoch{}.state'.format(epoch), optimizer)

    chainer.serializers.save_npz('classifier_final.model', model)
    chainer.serializers.save_npz('classifier_final.state', optimizer)
    with open('classifier_train_log.json', 'w') as fp:
        json.dump({'epoch': str(args.n_epoch), 'batch_size': str(args.batch_size), 'logs': logs},
            fp, sort_keys=True, ensure_ascii=False, indent=2)

def parse_arguments():
    usage = 'make training dataset catalog (sample code)'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--action', '-a', type=str, dest='action', required=True)
    # prepare options
    parser.add_argument('--train-catalog-file', type=str, dest='train_catalog_file', default='')
    parser.add_argument('--cv-catalog-file', type=str, dest='cv_catalog_file', default='')
    # train options
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
