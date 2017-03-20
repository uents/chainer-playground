# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Function, Link

# configurations
xp = np
N_BOXES = 1
N_CLASSES = 25  # 1..25
N_GRID = 7
INPUT_SIZE = 448
N_CNN_LAYER = 7


'''
yolo-tinyについては、Darknetの以下のリンク先を参考に実装。tiny-yoloとは異なるので注意
https://github.com/pjreddie/darknet/blob/8f1b4e0962857d402f9d017fcbf387ef0eceb7c4/cfg/yolo-tiny.cfg
'''

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
        batch_size = x.data.shape[0]
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
        # reshape result tensor
        h = F.reshape(h, (batch_size, -1))
        return h

    def __call__(self, x, t):
        h = self.forward(x)
        # ラベルの識別子は0始まりにしないとエラーするため-1する
        self.loss = F.softmax_cross_entropy(h, t-1)
        self.accuracy = F.accuracy(h, t)
        if self.train:
            return self.loss
        else:
            return F.softmax(h)

class YoloTiny(chainer.Chain):
    def __init__(self):
        super(YoloTiny, self).__init__(
            conv1  = L.Convolution2D(3,      16, ksize=3, stride=1, pad=1),
            conv2  = L.Convolution2D(None,   32, ksize=3, stride=1, pad=1),
            conv3  = L.Convolution2D(None,   64, ksize=3, stride=1, pad=1),
            conv4  = L.Convolution2D(None,  128, ksize=3, stride=1, pad=1),
            conv5  = L.Convolution2D(None,  256, ksize=3, stride=1, pad=1),
            conv6 = L.Convolution2D(None,  512, ksize=3, stride=1, pad=1),
            conv7 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            conv8 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            conv9 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            fc1 = L.Linear(50176, 256), # (1024,7,7)=50176
            fc2 = L.Linear(None, 4096),
            fc3 = L.Linear(None, ((N_BOXES*5)+N_CLASSES) * (N_GRID**2))
        )
        self.train = True

    def forward(self, x):
        batch_size = x.data.shape[0]
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
        h = F.leaky_relu(self.conv8(h), slope=0.1)
        h = F.leaky_relu(self.conv9(h), slope=0.1)
        h = F.leaky_relu(self.fc1(h), slope=0.1)
        h = F.leaky_relu(self.fc2(h), slope=0.1)
        # skip dropout
        h = self.fc3(h) # (batch_size, 1470)

        # extract result tensors
        h = F.reshape(h, (batch_size, (5*N_BOXES)+N_CLASSES, N_GRID, N_GRID))
        x, y, w, h, conf, prob = F.split_axis(h, indices_or_sections=(1,2,3,4,5), axis=1)
        # activation
        return F.sigmoid(x), F.sigmoid(y), F.sigmoid(w), F.sigmoid(h), F.sigmoid(conf), F.sigmoid(prob)

    def __call__(self, x, t):
        # 推論を実行
        px, py, pw, ph, pconf, pprob = self.forward(x)
        # 教師データを抽出
        tx, ty, tw, th, tconf, tprob = np.array_split(t.data, indices_or_sections=(1,2,3,4,5), axis=1)
        # オブジェクトが存在しないグリッドは、活性化後にグリッド中心となるよう学習
        tx[tconf == 0.0] = 0.5
        ty[tconf == 0.0] = 0.5
        # bounding boxの学習係数を、オブジェクトが存在するグリッドか否かで調整
        box_learning_scale = np.tile(0.1, tconf.shape)
        box_learning_scale[tconf == 1.0] = 1.0
        # confidencedの学習係数を、オブジェクトが存在するグリッドか否かで調整
        conf_learning_scale = np.tile(0.1, tconf.shape)
        conf_learning_scale[tconf == 1.0] = 10.0

        # 損失誤差を算出
        tx = self.__variable(tx, np.float32)
        ty = self.__variable(ty, np.float32)
        tw = self.__variable(tw, np.float32)
        th = self.__variable(th, np.float32)
        tconf = self.__variable(tconf, np.float32)
        tprob = self.__variable(tprob, np.float32)
        box_learning_scale = self.__variable(box_learning_scale, np.float32)
        conf_learning_scale = self.__variable(conf_learning_scale, np.float32)

        x_loss = F.sum(box_learning_scale * ((tx - px) ** 2)) / 2
        y_loss = F.sum(box_learning_scale * ((ty - py) ** 2)) / 2
        w_loss = F.sum(box_learning_scale * ((tw - pw) ** 2)) / 2
        h_loss = F.sum(box_learning_scale * ((th - ph) ** 2)) / 2
        conf_loss = F.sum(conf_learning_scale * ((tconf - pconf) ** 2)) / 2
        prob_loss = F.sum(box_learning_scale * F.reshape(F.sum(((tprob - pprob) ** 2), axis=1), box_learning_scale.shape)) / 2
        self.loss = x_loss + y_loss + w_loss + h_loss + conf_loss + prob_loss

        if self.train:
            print("loss x:%f y:%f w:%f h:%f conf:%f prob:%f" %
                (F.sum(x_loss).data, F.sum(y_loss).data, F.sum(w_loss).data, F.sum(h_loss).data, \
                F.sum(conf_loss).data, F.sum(prob_loss).data))
            return self.loss
        else:
            # TODO: 推論の実行
            return None

    def __variable(self, v, t):
        return chainer.Variable(xp.asarray(v).astype(t))
