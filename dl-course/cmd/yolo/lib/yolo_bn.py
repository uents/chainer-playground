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


'''
YOLO(YOLO Tiny)については、Darknetの以下のリンク先を参考に実装。Tiny YOLOとは異なるので注意
https://github.com/pjreddie/darknet/blob/8f1b4e0962857d402f9d017fcbf387ef0eceb7c4/cfg/yolo-tiny.cfg
'''

class YoloClassifier(chainer.Chain):
    def __init__(self, gpu=-1):
        super(YoloClassifier, self).__init__(
#            conv1  = L.Convolution2D(3,      16, ksize=3, stride=1, pad=1),

            # convolution layers
            conv1  = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1, nobias=True),
            bn1    = L.BatchNormalization(32, use_beta=False, eps=2e-5),
            bias1  = L.Bias(shape=(32,)),
            conv2  = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, nobias=True),
            bn2    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias2  = L.Bias(shape=(64,)),
            conv3  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn3    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias3  = L.Bias(shape=(128,)),
            conv4  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn4    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias4  = L.Bias(shape=(256,)),
            conv5  = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn5    = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias5  = L.Bias(shape=(512,)),
            conv6  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn6    = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias6  = L.Bias(shape=(1024,)),

            # additonal layers for pretraining
            conv7  = L.Convolution2D(None, N_CLASSES, ksize=1, stride=1, pad=0),
        )
        self.gpu = -1
        if gpu >= 0:
            self.gpu = gpu
            self.to_gpu()
        self.train = False

    def forward(self, x):
        batch_size = x.data.shape[0]

        # convolution layers
        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 112
        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 56
        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 28
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 14
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 7
        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h), test=not self.train)), slope=0.1)

        # additional layers for pretraining
        h = self.conv7(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)

        # reshape result tensor
        h = F.reshape(h, (batch_size, -1))
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
#            conv1  = L.Convolution2D(3,      16, ksize=3, stride=1, pad=1),

            conv1  = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1, nobias=True),
            bn1    = L.BatchNormalization(32, use_beta=False, eps=2e-5),
            bias1  = L.Bias(shape=(32,)),
            conv2  = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, nobias=True),
            bn2    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias2  = L.Bias(shape=(64,)),
            conv3  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn3    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias3  = L.Bias(shape=(128,)),
            conv4  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn4    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias4  = L.Bias(shape=(256,)),
            conv5  = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn5    = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias5  = L.Bias(shape=(512,)),
            conv6  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn6    = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias6  = L.Bias(shape=(1024,)),
            conv7  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn7    = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias7  = L.Bias(shape=(1024,)),
            conv8  = L.Convolution2D(3072, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn8    = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias8  = L.Bias(shape=(1024,)),

            conv9  = L.Convolution2D(None, (N_BOXES*5)+N_CLASSES, ksize=3, stride=1, pad=1, nobias=True),
            bias9  = L.Bias(shape=((N_BOXES*5)+N_CLASSES,)),
#            fc9    = L.Linear(50176, 2048), # (1024,7,7)=50176
#            fc10   = L.Linear(2048, ((N_BOXES*5)+N_CLASSES) * (N_GRID**2)),
        )
        self.gpu = -1
        if gpu >= 0:
            self.gpu = gpu
            self.to_gpu()
        self.train = False

    def forward(self, x):
        batch_size = x.data.shape[0]

        # convolution layers
        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 112

        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 56

        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 28

        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 14

        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h), test=not self.train)), slope=0.1)
        high_res_feature = reorg(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 7

        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h), test=not self.train)), slope=0.1)
        h = F.concat((high_res_feature, h), axis=1) # output concatination
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h), test=not self.train)), slope=0.1)
        h = self.bias9(self.conv9(h))

        # fully connection layers
#        h = F.leaky_relu(self.fc9(h), slope=0.1)
#        h = F.dropout(h, train=self.train, ratio=DROPOUT_RATIO)
#        h = F.leaky_relu(self.fc10(h), slope=0.1)

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

        self.loss = x_loss + y_loss + w_loss + h_loss + conf_loss + prob_loss
        self.loss_log = ("loss %3.4f x:%03.4f y:%03.4f w:%03.4f h:%03.4f conf:%03.4f prob:%03.4f" %
                         (self.loss.data, x_loss.data / batch_size, y_loss.data / batch_size,
                          w_loss.data / batch_size, h_loss.data / batch_size,
                          conf_loss.data / batch_size, prob_loss.data / batch_size))

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


def reorg(input, stride=2):
    batch_size, in_channel, in_height, in_width = input.data.shape
    out_height, out_width, out_channel \
        = int(in_height/stride), int(in_width/stride), in_channel*stride*stride
    output = F.transpose(F.reshape(input, (batch_size, in_channel, out_height, stride, out_width, stride)), (0, 1, 2, 4, 3, 5)) 
    output = F.transpose(F.reshape(output, (batch_size, in_channel, out_height, out_width, -1)), (0, 4, 1, 2, 3)) 
    output = F.reshape(output, (batch_size, out_channel, out_height, out_width))
    return output
