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


# file paths
TRAIN_DATASET_PATH = os.path.join('.', 'train_dataset.npz')

# configurations
xp = np
N_BOXES = 1
N_CLASSES = 25  # 1..25
N_GRID = 7
INPUT_SIZE = 448

class YoloTinyCNN(chainer.Chain):
    def __init__(self):
        super(YoloTinyCNN, self).__init__(
            conv1  = L.Convolution2D(3,      16, ksize=3, stride=1, pad=1),
            conv2  = L.Convolution2D(None,   32, ksize=3, stride=1, pad=1),
            conv3  = L.Convolution2D(None,   64, ksize=3, stride=1, pad=1),
            conv4  = L.Convolution2D(None,  128, ksize=3, stride=1, pad=1),
            conv5  = L.Convolution2D(None,  256, ksize=3, stride=1, pad=1),
            conv6  = L.Convolution2D(None,  512, ksize=3, stride=1, pad=1),
            conv7  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            # addditonal layers for pretraining
            conv8  = L.Convolution2D(None, N_CLASSES, ksize=1, stride=1, pad=0),
        )
        self.train = True

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 224x224
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 112x112
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 56x56
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 28x28
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 14x14
        h = F.leaky_relu(self.conv6(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 7x7
        h = F.leaky_relu(self.conv7(h), slope=0.1)
        # additional layers for pretraining
        h = self.conv8(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)
        return h

    def __call__(self, x, t):
        batch_size = x.data.shape[0]
        h = self.forward(x)
        h = F.reshape(h, (batch_size, -1))

        # ラベルの識別子は0始まりにしないとエラーするため-1する
        self.loss = F.softmax_cross_entropy(h, t-1)
        self.accuracy = F.accuracy(h, t)
        if self.train:
            return self.loss
        else:
            return F.softmax(h)


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

def prepare_dataset(catalog_path):
    with open(os.path.join(catalog_path), 'r') as fp:
        catalog = json.load(fp)
    train_dataset = catalog['dataset']
    inputs = np.array([load_image(item['color_image_path']) for item in train_dataset])
    labels = np.array([item['classes'] for item in train_dataset])
    np.savez(TRAIN_DATASET_PATH, inputs=inputs, labels=labels)


def one_epoch_train(model, optimizer, batch_size, inputs, labels):
    n_train = len(labels)
    perm = np.random.permutation(n_train)

    sum_loss, sum_acc = (0., 0.)
    for count in six.moves.range(0, n_train, batch_size):
        ix = perm[count:count+batch_size]
        xs = chainer.Variable(xp.asarray(inputs[ix]).astype(np.float32).transpose(0,3,1,2))
        ts = chainer.Variable(xp.asarray(labels[ix].ravel()).astype(np.int32))

        model.train = True
        optimizer.update(model, xs, ts)
        print('mini-batch:{} loss:{} acc:{}'.format((count/batch_size)+1, model.loss.data, model.accuracy.data))
        sum_loss += model.loss.data/len(ix)
        sum_acc += model.accuracy.data/len(ix)

    return sum_loss, sum_acc

def one_epoch_cv(model, optimizer, inputs, labels):
    xs = chainer.Variable(xp.asarray(inputs).astype(np.float32).transpose(0,3,1,2))
    ts = chainer.Variable(xp.asarray(labels.ravel()).astype(np.int32))

    model.train = False
    model(xs, ts)
    return model.loss.data, model.accuracy.data

def train_cnn(n_epoch, batch_size):
    model = YoloTinyCNN()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    logs = []
    train_dataset = np.load(TRAIN_DATASET_PATH)
    inputs = train_dataset['inputs']
    labels = train_dataset['labels']

    train_ixs = sorted(random.sample(range(labels.shape[0]), int(labels.shape[0] * 0.8)))
    cv_ixs = sorted(list(set(range(labels.shape[0])) - set(train_ixs)))
    print('number of dataset: train:{} cv:{}'.format(len(train_ixs), len(cv_ixs)))

    for epoch in six.moves.range(1, n_epoch+1):
        train_loss, train_acc = one_epoch_train(model, optimizer, batch_size, inputs[train_ixs], labels[train_ixs])
        cv_loss, cv_acc = one_epoch_cv(model, optimizer, inputs[cv_ixs], labels[cv_ixs])
        print('epoch:{} trian loss:{} train acc:{} cv loss:{} cv acc:{}'.format(epoch, train_loss, train_acc, cv_loss, cv_acc))
        logs.append({'epoch': str(epoch),
            'train_loss': str(train_loss), 'train_acc': str(train_acc),
            'cv_loss': str(cv_loss), 'cv_acc': str(cv_acc)})
        if (epoch % 10) == 0:
            chainer.serializers.save_npz('yolo-tiny-cnn_model_epoch{}.model'.format(epoch), model)
            chainer.serializers.save_npz('yolo-tiny-cnn_model_epoch{}.state'.format(epoch), optimizer)

    with open('yolo-tiny-cnn_train_log.json', 'w') as fp:
        json.dump({'epoch': str(n_epoch), 'batch_size': str(batch_size), 'logs': logs},
            fp, sort_keys=True, ensure_ascii=False, indent=2)

def parse_arguments():
    usage = 'make training dataset catalog (sample code)'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--action', '-a', type=str, dest='action', required=True)
    parser.add_argument('--catalog-file', type=str, dest='catalog_file')
    parser.add_argument('--batchsize', '-b', type=int, dest='batch_size', default=50)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--epoch', '-e', type=int, dest='n_epoch', default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
        xp = chainer.cuda.cupy

    if args.action == 'prepare':
        print('preparing: catalog={}'.format(args.catalog_file))
        prepare_dataset(args.catalog_file)
    elif args.action == 'train':
        print('training: gpu={} epoch={} batchsize={}'.format(args.gpu, args.n_epoch, args.batch_size))
        train_cnn(args.n_epoch, args.batch_size)

    print('\ndone')
