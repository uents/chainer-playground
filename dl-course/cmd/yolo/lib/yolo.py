# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import itertools
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from bounding_box import *

xp = np

'''
YOLO(YOLO Tiny)については、Darknetの以下のリンク先を参考に実装。Tiny YOLOとは異なるので注意
https://github.com/pjreddie/darknet/blob/8f1b4e0962857d402f9d017fcbf387ef0eceb7c4/cfg/yolo-tiny.cfg
'''

class YoloClassifier(chainer.Chain):
    def __init__(self, gpu=-1):
        super(YoloClassifier, self).__init__(
#            conv1  = L.Convolution2D(3,      16, ksize=3, stride=1, pad=1),
            conv1  = L.Convolution2D(None,   32, ksize=3, stride=1, pad=1),
            conv2  = L.Convolution2D(None,   64, ksize=3, stride=1, pad=1),
            conv3  = L.Convolution2D(None,  128, ksize=3, stride=1, pad=1),
            conv4  = L.Convolution2D(None,  256, ksize=3, stride=1, pad=1),
            conv5  = L.Convolution2D(None,  512, ksize=3, stride=1, pad=1),
            conv6  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            # additonal layers for pretraining
            conv7  = L.Convolution2D(None, N_CLASSES, ksize=1, stride=1, pad=0),
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
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 112
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 56
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 28
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 14
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 7
        h = F.leaky_relu(self.conv6(h), slope=0.1)

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
            conv1  = L.Convolution2D(None,   32, ksize=3, stride=1, pad=1),
            conv2  = L.Convolution2D(None,   64, ksize=3, stride=1, pad=1),
            conv3  = L.Convolution2D(None,  128, ksize=3, stride=1, pad=1),
            conv4  = L.Convolution2D(None,  256, ksize=3, stride=1, pad=1),
            conv5  = L.Convolution2D(None,  512, ksize=3, stride=1, pad=1),
            conv6  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            conv7  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            conv8  = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1),
            fc1  = L.Linear(50176, 4096), # (1024,7,7)=50176
            fc2  = L.Linear(None, ((N_BOXES*5)+N_CLASSES) * (N_GRID**2))
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
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 112
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 56
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 28
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 14
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0) # 7
        h = F.leaky_relu(self.conv6(h), slope=0.1)
        h = F.leaky_relu(self.conv7(h), slope=0.1)
        h = F.leaky_relu(self.conv8(h), slope=0.1)

        # fully connection layers
        h = F.leaky_relu(self.fc1(h), slope=0.1)
        h = F.dropout(h, train=self.train, ratio=DROPOUT_RATIO)
        h = self.fc2(h)

        # normalize and reshape predicted tensors
        h = F.sigmoid(h)
        h = F.reshape(h, (batch_size, (5*N_BOXES)+N_CLASSES, N_GRID, N_GRID))
        return h

    def __call__(self, x, t, debug=False):
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

        self.h = self.from_variable(h)

        n_correct = 0
        positives = init_positives()
        for batch in six.moves.range(0, batch_size):
            truth_boxes = [box['box'] for box in decode_box_tensor(t.data.get()[batch])]
            predicted_boxes = self.tensor_to_boxes(self.h[batch])
            positives = add_positives(positives, count_positives(predicted_boxes, truth_boxes))
            for pred_box, truth_box in itertools.product(predicted_boxes, truth_boxes):
                correct, iou = Box.correct(pred_box, [truth_box])
#                print('{0} {1} {2:.3f} pred:{3} truth:{4}'.format(
#                    batch + 1, correct, iou, pred_box, truth_box))
                n_correct += int(correct)

        n_correct = max(n_correct, 1)
        self.mean_ap = mean_average_precision(positives)

        self.loss_log = ("loss corr:%d x:%3.4f y:%3.4f w:%3.4f h:%3.4f conf:%3.4f prob:%3.4f" %
                         (n_correct, x_loss.data / batch_size, y_loss.data / batch_size,
                          w_loss.data / batch_size, h_loss.data / batch_size,
                          conf_loss.data / batch_size, prob_loss.data / batch_size))
        self.loss = (x_loss + y_loss + w_loss + h_loss + conf_loss + prob_loss) / n_correct
        if self.train:
            return self.loss
        else:
            return self.h

    def predict(self, x):
        h = self.forward(x)
        return self.from_variable(h)

    def tensor_to_boxes(self, tensor):
        return nms(select_candidates(tensor))

    def from_variable(self, v):
        return chainer.cuda.to_cpu(v.data)

    def to_variable(self, v, t=np.float32):
        v = v.astype(t)
        if self.gpu >= 0:
            v = chainer.cuda.to_gpu(v)
        return chainer.Variable(v)


def init_positives():
    return [{'true': 0, 'false': 0}  for i in range(0, N_CLASSES)]

def count_positives(predicted_boxes, truth_boxes):
    positives = init_positives()
    for pred_box in predicted_boxes:
        correct, iou = Box.correct(pred_box, truth_boxes)
        if correct:
            positives[int(pred_box.clazz)]['true'] += 1
        else:
            positives[int(pred_box.clazz)]['false'] += 1
    return positives

def add_positives(pos1, pos2):
    def add_item(item1, item2):
        return {'true': item1['true'] + item2['true'],
                'false': item1['false'] + item2['false']}
    return [add_item(item1, item2) for item1, item2 in zip(pos1, pos2)]

def average_precisions(positives):
    def precision(tp, fp):
        if tp == 0 and fp == 0:
            return 0.
        return float(tp) / (tp + fp)
    return [precision(p['true'], p['false']) for p in positives]

def mean_average_precision(positives):
    aps = average_precisions(positives)
    return np.asarray(aps).mean()
