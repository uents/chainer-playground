
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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from bounding_box import *

xp = np

class YoloClassifier(chainer.Chain):
    def __init__(self, gpu=-1):
        super(YoloClassifier, self).__init__(
            conv1  = L.Convolution2D(3,   64,  ksize=7, stride=2, pad=3),

            conv2  = L.Convolution2D(64,  192, ksize=3, stride=2, pad=1),

            conv3  = L.Convolution2D(192, 128, ksize=1, stride=1, pad=1),
            conv4  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            conv5  = L.Convolution2D(256, 256, ksize=1, stride=1, pad=1),
            conv6  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),

            conv7  = L.Convolution2D(512, 256, ksize=1, stride=1, pad=1),
            conv8  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv9  = L.Convolution2D(512, 256, ksize=1, stride=1, pad=1),
            conv10 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv11 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=1),
            conv12 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv13 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=1),
            conv14 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv15 = L.Convolution2D(512, 512, ksize=1, stride=1, pad=1),
            conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),

            conv17 = L.Convolution2D(1024, 512,  ksize=1, stride=1, pad=1),
            conv18 = L.Convolution2D(512,  1024, ksize=3, stride=1, pad=1),
            conv19 = L.Convolution2D(1024, 512,  ksize=1, stride=1, pad=1),
            conv20 = L.Convolution2D(512,  1024, ksize=3, stride=1, pad=1),

            # additonal layers for pretraining
#            conv_pre = L.Convolution2D(None, N_CLASSES, ksize=1, stride=1, pad=0),
            fc_pre = L.Linear(1024, N_CLASSES)
        )
        self.gpu = -1
        if gpu >= 0:
            xp = chainer.cuda.cupy
            self.gpu = gpu
            self.to_gpu()
        self.train = False

    def forward(self, x):
        batch_size = x.data.shape[0]

        # convolution layers
        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = F.leaky_relu(self.conv6(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.conv7(h), slope=0.1)
        h = F.leaky_relu(self.conv8(h), slope=0.1)
        h = F.leaky_relu(self.conv9(h), slope=0.1)
        h = F.leaky_relu(self.conv10(h), slope=0.1)
        h = F.leaky_relu(self.conv11(h), slope=0.1)
        h = F.leaky_relu(self.conv12(h), slope=0.1)
        h = F.leaky_relu(self.conv13(h), slope=0.1)
        h = F.leaky_relu(self.conv14(h), slope=0.1)
        h = F.leaky_relu(self.conv15(h), slope=0.1)
        h = F.leaky_relu(self.conv16(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.conv17(h), slope=0.1)
        h = F.leaky_relu(self.conv18(h), slope=0.1)
        h = F.leaky_relu(self.conv19(h), slope=0.1)
        h = F.leaky_relu(self.conv20(h), slope=0.1)

        # additional layers for pretraining
#        print(h.shape)
#        h = self.conv_pre(h)
#        print(h.shape)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)
#        print(h.shape)
        h = self.fc_pre(h)

        # reshape result tensor
#        print(h.shape)
        h = F.reshape(h, (batch_size, -1))
#        print(h.shape)
        return h

    def __call__(self, x, t):
        h = self.forward(x)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        if self.train:
            return self.loss
        else:
            return F.softmax(h)

class YoloDetector(chainer.Chain):
    def __init__(self, gpu=-1):
        super(YoloDetector, self).__init__(
            conv1  = L.Convolution2D(3,   64,  ksize=7, stride=2, pad=3),

            conv2  = L.Convolution2D(64,  192, ksize=3, stride=2, pad=1),

            conv3  = L.Convolution2D(192, 128, ksize=1, stride=1, pad=1),
            conv4  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            conv5  = L.Convolution2D(256, 256, ksize=1, stride=1, pad=1),
            conv6  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),

            conv7  = L.Convolution2D(512, 256, ksize=1, stride=1, pad=1),
            conv8  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv9  = L.Convolution2D(512, 256, ksize=1, stride=1, pad=1),
            conv10 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv11 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=1),
            conv12 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv13 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=1),
            conv14 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv15 = L.Convolution2D(512, 512, ksize=1, stride=1, pad=1),
            conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),

            conv17 = L.Convolution2D(1024, 512,  ksize=1, stride=1, pad=1),
            conv18 = L.Convolution2D(512,  1024, ksize=3, stride=1, pad=1),
            conv19 = L.Convolution2D(1024, 512,  ksize=1, stride=1, pad=1),
            conv20 = L.Convolution2D(512,  1024, ksize=3, stride=1, pad=1), 
            conv21 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1),
            conv22 = L.Convolution2D(1024, 1024, ksize=3, stride=2, pad=1),
            conv23 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1),
            conv24 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1),

            fc25 = L.Linear(50176, 4096),
            fc26 = L.Linear(4096, ((N_BOXES*5)+N_CLASSES) * (N_GRID**2))
        )
        self.gpu = -1
        if gpu >= 0:
            xp = chainer.cuda.cupy
            self.gpu = gpu
            self.to_gpu()
        self.train = False

    def forward(self, x):
        batch_size = x.data.shape[0]

        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = F.leaky_relu(self.conv6(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.conv7(h), slope=0.1)
        h = F.leaky_relu(self.conv8(h), slope=0.1)
        h = F.leaky_relu(self.conv9(h), slope=0.1)
        h = F.leaky_relu(self.conv10(h), slope=0.1)
        h = F.leaky_relu(self.conv11(h), slope=0.1)
        h = F.leaky_relu(self.conv12(h), slope=0.1)
        h = F.leaky_relu(self.conv13(h), slope=0.1)
        h = F.leaky_relu(self.conv14(h), slope=0.1)
        h = F.leaky_relu(self.conv15(h), slope=0.1)
        h = F.leaky_relu(self.conv16(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.conv17(h), slope=0.1)
        h = F.leaky_relu(self.conv18(h), slope=0.1)
        h = F.leaky_relu(self.conv19(h), slope=0.1)
        h = F.leaky_relu(self.conv20(h), slope=0.1)
        h = F.leaky_relu(self.conv21(h), slope=0.1)
        h = F.leaky_relu(self.conv22(h), slope=0.1)
        h = F.leaky_relu(self.conv23(h), slope=0.1)
        h = F.leaky_relu(self.conv24(h), slope=0.1)
#        print(h.shape)

        # fully connection layers
        h = self.fc25(h)
        h = F.dropout(h, train=self.train, ratio=DROPOUT_RATIO)        
        h = F.leaky_relu(h, slope=0.1)
        h = self.fc26(h)

        # normalize and reshape predicted tensors
        h = F.sigmoid(h)
        h = F.reshape(h, (batch_size, (5*N_BOXES)+N_CLASSES, N_GRID, N_GRID))
        return h

    def __call__(self, x, t):
        batch_size = t.data.shape[0]

        # 推論を実行
        h = self.forward(x)
        px, py, pw, ph, pconf, pprob \
            = F.split_axis(h, indices_or_sections=(1,2,3,4,5), axis=1)
        # 教師データを抽出
        tx, ty, tw, th, tconf, tprob \
            = np.array_split(self.from_variable(t), indices_or_sections=(1,2,3,4,5), axis=1)

        # オブジェクトが存在しないグリッドは、グリッド中心とする
        tx[tconf != 1.] = 0.5
        ty[tconf != 1.] = 0.5
        # オブジェクトが存在しないグリッドは、学習させない(誤差を相殺する)
        class_map = (tprob == 1.0)
        tprob = self.from_variable(pprob)
        tprob[class_map] = 1.0

        # 学習係数を、オブジェクトが存在するグリッドか否かで調整
        box_scale_factor = np.tile(SCALE_FACTORS['nocoord'], tconf.shape).astype(np.float32)
        box_scale_factor[tconf == 1.0] = SCALE_FACTORS['coord']
        conf_scale_factor = np.tile(SCALE_FACTORS['noobj'], tconf.shape).astype(np.float32)
        conf_scale_factor[tconf == 1.0] = SCALE_FACTORS['obj']
        prob_scale_factor = np.tile(0.0, tconf.shape).astype(np.float32)
        prob_scale_factor[tconf == 1.0] = SCALE_FACTORS['prob']

        # 損失誤差を算出
        tx, ty, tw, th = self.to_variable(tx), self.to_variable(ty), self.to_variable(tw), self.to_variable(th)
        tconf, tprob = self.to_variable(tconf), self.to_variable(tprob)
        box_scale_factor = self.to_variable(box_scale_factor)
        conf_scale_factor = self.to_variable(conf_scale_factor)
        prob_scale_factor = self.to_variable(prob_scale_factor)

        x_loss = F.sum(box_scale_factor * ((tx - px) ** 2))
        y_loss = F.sum(box_scale_factor * ((ty - py) ** 2))
        w_loss = F.sum(box_scale_factor * ((tw - pw) ** 2))
        h_loss = F.sum(box_scale_factor * ((th - ph) ** 2))
        conf_loss = F.sum(conf_scale_factor * ((tconf - pconf) ** 2))
        prob_loss = F.sum(prob_scale_factor * F.reshape(F.sum(((tprob - pprob) ** 2), axis=1), prob_scale_factor.shape))

        self.loss_log = ("loss x:%03.4f y:%03.4f w:%03.4f h:%03.4f conf:%03.4f prob:%03.4f" %
                         (x_loss.data / batch_size, y_loss.data / batch_size,
                          w_loss.data / batch_size, h_loss.data / batch_size,
                          conf_loss.data / batch_size, prob_loss.data / batch_size))
        self.loss = x_loss + y_loss + w_loss + h_loss + conf_loss + prob_loss

        self.h = self.from_variable(h)
        if self.train:
            return self.loss
        else:
            return self.h

    def predict(self, x):
        h = self.forward(x)
        return self.from_variable(h)

    def from_variable(self, v):
        return chainer.cuda.to_cpu(v.data)

    def to_variable(self, v):
        v = v.astype(np.float32)
        if self.gpu >= 0:
            v = chainer.cuda.to_gpu(v)
        return chainer.Variable(v)
