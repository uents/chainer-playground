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

N_CNN_LAYER = 6

class YoloClassifier(chainer.Chain):
    def __init__(self, gpu=-1):
        super(YoloClassifier, self).__init__(
            conv1  = L.Convolution2D(3,      32, ksize=3, stride=1, pad=1),
            conv2  = L.Convolution2D(None,   64, ksize=3, stride=1, pad=1),
            conv3  = L.Convolution2D(None,  128, ksize=3, stride=1, pad=1),
            conv4  = L.Convolution2D(None,  256, ksize=3, stride=1, pad=1),
            conv5  = L.Convolution2D(None,  512, ksize=3, stride=1, pad=1),
            conv6  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            # additonal layer for pre-training
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
        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv6(h), slope=0.1)

        # additional layer for pre-training
        h = self.conv7(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)

        # reshape output tensor
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
    def __init__(self, gpu=-1, n_grid=N_GRID, anchor_boxes=ANCHOR_BOXES):
        self.n_grid = n_grid
        self.anchor_boxes = anchor_boxes # YOLOv2との互換性のため便宜的に持つ
        
        super(YoloDetector, self).__init__(
            conv1  = L.Convolution2D(None,   32, ksize=3, stride=1, pad=1),
            conv2  = L.Convolution2D(None,   64, ksize=3, stride=1, pad=1),
            conv3  = L.Convolution2D(None,  128, ksize=3, stride=1, pad=1),
            conv4  = L.Convolution2D(None,  256, ksize=3, stride=1, pad=1),
            conv5  = L.Convolution2D(None,  512, ksize=3, stride=1, pad=1),
            conv6  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            conv7  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            conv8  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            fc9    = L.Linear(50176, 4096), # 50176=1024x7x7
            fc10   = L.Linear(4096, (5+N_CLASSES) * (self.n_grid**2))
        )
        self.gpu = -1
        if gpu >= 0:
            self.gpu = gpu
            self.to_gpu()
        self.train = False
        self.iter_count = 1

    def forward(self, x):
        batch_size = x.data.shape[0]

        # convolution layers
        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.conv6(h), slope=0.1)
        h = F.leaky_relu(self.conv7(h), slope=0.1)
        h = F.leaky_relu(self.conv8(h), slope=0.1)

        # full connection layers
        h = F.leaky_relu(self.fc9(h), slope=0.1)
        h = F.dropout(h, train=self.train, ratio=DROPOUT_RATIO)
        h = F.leaky_relu(self.fc10(h), slope=0.1)

        # reshape output tensor
        h = F.reshape(h, (batch_size, 1, 5+N_CLASSES, self.n_grid, self.n_grid))
        h = F.sigmoid(h)
        return h

    def __call__(self, x, ground_truths):
        batch_size = x.data.shape[0]

        # 推論を実行し結果を抽出
        h = self.forward(x)
        px, py, pw, ph, pconf, pprob \
            = F.split_axis(h, indices_or_sections=(1,2,3,4,5), axis=2)

        # 教師データを初期化
        tx = np.tile(0.5, px.shape).astype(np.float32) # 基本は0.5 (グリッド中心)
        ty = np.tile(0.5, py.shape).astype(np.float32) # 基本は0.5 (グリッド中心)
        tw = np.zeros(pw.shape).astype(np.float32)
        th = np.zeros(ph.shape).astype(np.float32)
        tconf = np.zeros(pconf.shape).astype(np.float32)
        tprob = pprob.data.copy() # 真のグリッド以外は損失誤差が発生しないよう推定値をコピー

        # scaling factorを初期化
        box_scale_factor = np.tile(SCALE_FACTORS['nocoord'], tconf.shape).astype(np.float32)
        conf_scale_factor = np.tile(SCALE_FACTORS['noconf'], tconf.shape).astype(np.float32)

        # objectに最も近い教師データをground truthに近づける
        if self.iter_count >= 30:
            for batch in six.moves.range(0, batch_size):
                for truth_box in ground_truths[batch]:
                    grid_x = int(math.modf(truth_box.center.x)[1])
                    grid_y = int(math.modf(truth_box.center.y)[1])
                    tx[batch, :, :, grid_y, grid_x] = math.modf(truth_box.center.x)[0]
                    ty[batch, :, :, grid_y, grid_x] = math.modf(truth_box.center.y)[0]
                    tw[batch, :, :, grid_y, grid_x] = truth_box.width / self.n_grid
                    th[batch, :, :, grid_y, grid_x] = truth_box.height / self.n_grid
                    box_scale_factor[batch, :, :, grid_y, grid_x] = SCALE_FACTORS['coord']

                    # TODO: IOU^{truth}_{pred}とすべき??
                    tconf[batch, :, :, grid_y, grid_x] = 1.
                    conf_scale_factor[batch, :, :, grid_y, grid_x] = SCALE_FACTORS['conf']

                    tprob[batch, :, :, grid_y, grid_x] = 0.
                    tprob[batch, :, int(truth_box.clazz), grid_y, grid_x] = 1.

        
        # 損失誤差を算出
        tx, ty, tw, th = self.to_variable(tx), self.to_variable(ty), \
                         self.to_variable(tw), self.to_variable(th)
        tconf, tprob = self.to_variable(tconf), self.to_variable(tprob)
        box_scale_factor = self.to_variable(box_scale_factor)
        conf_scale_factor = self.to_variable(conf_scale_factor)

        loss_x = F.sum(box_scale_factor * ((px - tx) ** 2))
        loss_y = F.sum(box_scale_factor * ((py - ty) ** 2))
        loss_w = F.sum(box_scale_factor * ((pw - tw) ** 2))
        loss_h = F.sum(box_scale_factor * ((ph - th) ** 2))
        loss_conf = F.sum(conf_scale_factor * ((pconf - tconf) ** 2))
        loss_prob = F.sum((pprob - tprob) ** 2)

        self.loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_prob
        self.loss_log = ("loss %03.4f x:%03.4f y:%03.4f w:%03.4f h:%03.4f conf:%03.4f prob:%03.4f" %
                         (self.loss.data, loss_x.data / batch_size, loss_y.data / batch_size,
                          loss_w.data / batch_size, loss_h.data / batch_size,
                          loss_conf.data / batch_size, loss_prob.data / batch_size))

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
