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
from bounding_box import *

# configurations
xp = np
N_BOXES = 1
N_CLASSES = 26  # 0..25
                # F.softmax_cross_entropy()で扱うラベルが
                # 0始まりの必要があるため、便宜的に0を追加
N_GRID = 7
INPUT_SIZE = 448
N_CNN_LAYER = 7


'''
YOLO(YOLO Tiny)については、Darknetの以下のリンク先を参考に実装。Tiny YOLOとは異なるので注意
https://github.com/pjreddie/darknet/blob/8f1b4e0962857d402f9d017fcbf387ef0eceb7c4/cfg/yolo-tiny.cfg
'''

class YoloClassifier(chainer.Chain):
    def __init__(self, gpu=-1):
        super(YoloClassifier, self).__init__(
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
        self.train = False
        self.gpu = gpu
        if self.gpu >= 0: self.to_gpu()

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
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        if self.train:
            return self.loss
        else:
            return F.softmax(h)

class YoloDetector(chainer.Chain):
    def __init__(self, gpu=-1):
        super(YoloDetector, self).__init__(
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
        self.train = False
        self.class_prob_thresh = 0.3
        self.iou_thresh = 0.3
        self.gpu = gpu
        if self.gpu >= 0: self.to_gpu()

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
        h = F.dropout(h, train=self.train, ratio=0.5)
        h = F.leaky_relu(self.fc2(h), slope=0.1)
        h = self.fc3(h) # (batch_size, ((5*N_BOXES)+N_CLASSES)*N_GRID*N_GRID)

        # extract result tensors
        h = F.reshape(h, (batch_size, (5*N_BOXES)+N_CLASSES, N_GRID, N_GRID))
        x, y, w, h, conf, prob = F.split_axis(h, indices_or_sections=(1,2,3,4,5), axis=1)
        return F.sigmoid(x), F.sigmoid(y), F.sigmoid(w), F.sigmoid(h), F.sigmoid(conf), F.sigmoid(prob)

    def __call__(self, x, t):
        # 推論を実行
        px, py, pw, ph, pconf, pprob = self.forward(x)
        # 教師データを抽出
        if self.gpu >= 0: t.to_cpu()
        tx, ty, tw, th, tconf, _tprob = np.array_split(t.data, indices_or_sections=(1,2,3,4,5), axis=1)
#        tx, ty, tw, th, tconf, tprob = F.split_axis(t, indices_or_sections=(1,2,3,4,5), axis=1)
        if self.gpu >= 0: t.to_gpu()

        # オブジェクトが存在しないグリッドは、活性化後にグリッド中心となるよう学習
        tx[tconf != 1.0] = 0.5
        ty[tconf != 1.0] = 0.5
        # オブジェクトが存在しないグリッドは、学習させない
        if self.gpu >= 0: pprob.to_cpu()
        tprob = pprob.data.copy()
        if self.gpu >= 0: pprob.to_gpu()
        tprob[_tprob == 1.0] = 1.0
        # 学習係数を、オブジェクトが存在するグリッドか否かで調整
        box_learning_scale = np.tile(0.1, tconf.shape)
        box_learning_scale[tconf == 1.0] = 5.0
        conf_learning_scale = np.tile(0.5, tconf.shape)
        conf_learning_scale[tconf == 1.0] = 1.0
        prob_learning_scale = np.tile(0.0, tconf.shape)
        prob_learning_scale[tconf == 1.0] = 2.0

        # 損失誤差を算出
        tx = self.__variable(tx, np.float32)
        ty = self.__variable(ty, np.float32)
        tw = self.__variable(tw, np.float32)
        th = self.__variable(th, np.float32)
        tconf = self.__variable(tconf, np.float32)
        tprob = self.__variable(tprob, np.float32)
        if self.gpu >=0:
            tx.to_gpu(), ty.to_gpu(), tw.to_gpu(), th.to_gpu(), tconf.to_gpu(), tprob.to_gpu()
        box_learning_scale = self.__variable(box_learning_scale, np.float32)
        conf_learning_scale = self.__variable(conf_learning_scale, np.float32)
        prob_learning_scale = self.__variable(prob_learning_scale, np.float32)
        if self.gpu >=0:
            box_learning_scale.to_gpu(), conf_learning_scale.to_gpu(), prob_learning_scale.to_gpu()

        x_loss = F.sum(box_learning_scale * ((tx - px) ** 2))
        y_loss = F.sum(box_learning_scale * ((ty - py) ** 2))
        w_loss = F.sum(box_learning_scale * ((tw - pw) ** 2))
        h_loss = F.sum(box_learning_scale * ((th - ph) ** 2))
        conf_loss = F.sum(conf_learning_scale * ((tconf - pconf) ** 2))
        prob_loss = F.sum(prob_learning_scale * F.reshape(F.sum(((tprob - pprob) ** 2), axis=1), prob_learning_scale.shape))

        if self.train:
            print("loss x:%03.4f y:%03.4f w:%03.4f h:%03.4f conf:%03.4f prob:%03.4f" %
                  (x_loss.data, y_loss.data, w_loss.data, h_loss.data, conf_loss.data, prob_loss.data))
        self.loss = x_loss + y_loss + w_loss + h_loss + conf_loss + prob_loss

        if self.gpu >= 0:
            px.to_cpu(), py.to_cpu(), pw.to_cpu(), ph.to_cpu(), pconf.to_cpu(), pprob.to_cpu()
        self.detections = self.__detection(px, py, pw, ph, pconf, pprob)
        if self.gpu >= 0:
            px.to_gpu(), py.to_gpu(), pw.to_gpu(), ph.to_gpu(), pconf.to_gpu(), pprob.to_gpu()

        if self.train:
            return self.loss
        else:
            return self.detections

    def inference(self, x):
        px, py, pw, ph, pconf, pprob = self.forward(x)
        return self.__detection(px, py, pw, ph, pconf, pprob)

    def __detection(self, px, py, pw, ph, pconf, pprob):
        batch_size = px.data.shape[0]
        _px = F.reshape(px, (batch_size, N_GRID, N_GRID)).data
        _py = F.reshape(py, (batch_size, N_GRID, N_GRID)).data
        _pw = F.reshape(pw, (batch_size, N_GRID, N_GRID)).data
        _ph = F.reshape(ph, (batch_size, N_GRID, N_GRID)).data
        _pconf = F.reshape(pconf, (batch_size, N_GRID, N_GRID)).data
        _pprob = pprob.data

        detections = []
        for i in range(0, batch_size):
            candidates = self.__select_candidates(
                _px[i], _py[i], _pw[i], _ph[i], _pconf[i], _pprob[i], self.class_prob_thresh)
            winners = self.__nms(candidates, self.iou_thresh)
            detections.append(winners)
        return detections

    def __select_candidates(self, px, py, pw, ph, pconf, pprob, class_prob_thresh):
        class_prob_map = pprob * pconf # クラス確率を算出 (N_CLASSES,N_GRID,N_GRID)
        candidate_map = class_prob_map.max(axis=0) > class_prob_thresh # 検出グリッド候補を決定 (N_GRID,N_GRID)
        candidate_label_map = class_prob_map.argmax(axis=0) # 検出グリッド候補のラベルを抽出 (N_GRID,N_GRID)
        active_grid_cells = [{'x': float(point[1]), 'y': float(point[0])}
                             for point in np.argwhere(candidate_map == True)]

        candidates = []
        for i in range(0, candidate_map.sum()):
            candidates.append(
                Box(x=active_grid_cells[i]['x'] + px[candidate_map][i],
                    y=active_grid_cells[i]['y'] + py[candidate_map][i],
                    width=pw[candidate_map][i],
                    height=ph[candidate_map][i],
                    #'conf': pconf[candidate_map][i],
                    #'prob': pprob.transpose(1,2,0)[candidate_map][i],
                    clazz=candidate_label_map[candidate_map][i],
                    objectness=class_prob_map.max(axis=0)[candidate_map][i]
                ))
        return candidates

    def __nms(self, candidates, iou_thresh):
        sorted(candidates, key=lambda x: x.objectness, reverse=True)
        winners = []

        if len(candidates) == 0:
            return winners

        winners.append(candidates[0]) # 第１候補は必ず採用
        for i in range(1, len(candidates)): # 第２候補以降は上位の候補とのIOU次第
            for j in range(0, i):
                if Box.iou(candidates[i], candidates[j]) > iou_thresh:
                    break
            else:
                winners.append(candidates[i])
        return winners

    def __variable(self, v, t):
        return chainer.Variable(xp.asarray(v).astype(t))
