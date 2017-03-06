# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import sys
import os
import math
import random
import numpy as np
import scipy.io
#import seaborn as sns
#import matplotlib.pyplot as plt

import chainer
from chainer import Variable, Function, Link, optimizers
import chainer.functions as F
import chainer.links as L

from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

class MLP(chainer.Chain):
    def __init__(self, n_in, n_mid, n_out):
        super(MLP, self).__init__(
            fc_in  = L.Linear(n_in, n_mid),
            fc_mid = L.Linear(n_mid, n_mid),
            fc_out = L.Linear(n_mid, n_out)
        )
        self.train = True

    def __call__(self, x):
        h = F.sigmoid(self.fc_in(x))
        h = F.sigmoid(self.fc_mid(h))
        return self.fc_out(h)

def load_dataset():
    dataset = scipy.io.loadmat('./ex4data1.mat')
    xs = dataset['X'].tolist()
    ys = dataset['y'].ravel().tolist()

    # 元のラベルデータは0のラベルが10なので、0に置き換え直す
    ys = np.asarray([0 if y == 10 else y for y in ys]).reshape(len(ys)).astype(np.int32)
    # 画像データを (5000, 20, 20) に置き換え直す
    xs = np.asarray([np.asarray(x).reshape(20,20).T for x in xs]).reshape(len(ys), 20, 20).astype(np.float32)
    return xs, ys

def devide_dataset(xs, ys):
    # 学習サンプルを訓練データとテストデータに分割
    train_ixs = sorted(random.sample(range(ys.shape[0]), int(ys.shape[0] * 0.8)))
    test_ixs = sorted(list(set(range(ys.shape[0])) - set(train_ixs)))
    return xs[train_ixs, :], ys[train_ixs], xs[test_ixs, :], ys[test_ixs]


def main():
    # prepare dataset
    xs, ys = load_dataset()
    train_xs, train_ys, test_xs, test_ys = devide_dataset(xs, ys)

    # create learning model
    n_in = xs[0].shape[0] * xs[0].shape[1]
    n_mid = 25
    n_out = len(np.unique(ys))

    model = L.Classifier(MLP(n_in, n_mid, n_out))
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)

    # learning
    train = tuple_dataset.TupleDataset(train_xs, train_ys)
    test = tuple_dataset.TupleDataset(test_xs, test_ys)
    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (50, 'epoch'), out="result")

    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.dump_graph('main/loss')) # Windows環境ではなぜかKeyErrorが発生する...
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

if __name__ == '__main__':
    main()
