
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

N_CNN_LAYER = 18


class YoloClassifier(chainer.Chain):
    '''
    Darknet-19
    '''
    def __init__(self, gpu=-1):
        super(YoloClassifier, self).__init__(
            conv1  = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, nobias=True),
            bn1    = L.BatchNormalization(32, use_beta=False, eps=2e-5),
            bias1  = L.Bias(shape=(32,)),

            conv2  = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, nobias=True),
            bn2    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias2  = L.Bias(shape=(64,)),

            conv3  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn3    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias3  = L.Bias(shape=(128,)),
            conv4  = L.Convolution2D(None, 64, ksize=1, stride=1, pad=0, nobias=True),
            bn4    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias4  = L.Bias(shape=(64,)),
            conv5  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn5    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias5  = L.Bias(shape=(128,)),

            conv6  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn6    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias6  = L.Bias(shape=(256,)),
            conv7  = L.Convolution2D(None, 128, ksize=1, stride=1, pad=0, nobias=True),
            bn7    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias7  = L.Bias(shape=(128,)),
            conv8  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn8    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias8  = L.Bias(shape=(256,)),

            conv9  = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn9    = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias9  = L.Bias(shape=(512,)),
            conv10 = L.Convolution2D(None, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn10   = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias10 = L.Bias(shape=(256,)),
            conv11 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn11   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias11 = L.Bias(shape=(512,)),
            conv12 = L.Convolution2D(None, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn12   = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias12 = L.Bias(shape=(256,)),
            conv13 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn13   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias13 = L.Bias(shape=(512,)),

            conv14 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn14   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias14 = L.Bias(shape=(1024,)),
            conv15 = L.Convolution2D(None, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn15   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias15 = L.Bias(shape=(512,)),
            conv16 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn16   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias16 = L.Bias(shape=(1024,)),
            conv17 = L.Convolution2D(None, 512,  ksize=1, stride=1, pad=0, nobias=True),
            bn17   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias17 = L.Bias(shape=(512,)),
            conv18 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn18   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias18 = L.Bias(shape=(1024,)),

            # additonal layer for pretraining
            conv19 = L.Convolution2D(None, N_CLASSES, ksize=1, stride=1, pad=0)
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
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias9(self.bn9(self.conv9(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias10(self.bn10(self.conv10(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias11(self.bn11(self.conv11(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias12(self.bn12(self.conv12(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias13(self.bn13(self.conv13(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h), test=not self.train)), slope=0.1)

        # additional layer for pretraining
        h = self.conv19(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)

        # reshape output tensor
        h = F.reshape(h, (batch_size, -1))
        return h

    def __call__(self, x, t):
        h = self.forward(x)
        # TODO: use sum of square error as loss function
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        if self.train:
            return self.loss
        else:
            return F.softmax(h)


class YoloDetector(chainer.Chain):
    '''
    YOLO Detector
    '''
    def __init__(self, gpu=-1):
        super(YoloDetector, self).__init__(
            # common layers with Darknet-19
            conv1  = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, nobias=True),
            bn1    = L.BatchNormalization(32, use_beta=False, eps=2e-5),
            bias1  = L.Bias(shape=(32,)),

            conv2  = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, nobias=True),
            bn2    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias2  = L.Bias(shape=(64,)),

            conv3  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn3    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias3  = L.Bias(shape=(128,)),
            conv4  = L.Convolution2D(None, 64, ksize=1, stride=1, pad=0, nobias=True),
            bn4    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias4  = L.Bias(shape=(64,)),
            conv5  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn5    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias5  = L.Bias(shape=(128,)),

            conv6  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn6    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias6  = L.Bias(shape=(256,)),
            conv7  = L.Convolution2D(None, 128, ksize=1, stride=1, pad=0, nobias=True),
            bn7    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias7  = L.Bias(shape=(128,)),
            conv8  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn8    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias8  = L.Bias(shape=(256,)),

            conv9  = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn9    = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias9  = L.Bias(shape=(512,)),
            conv10 = L.Convolution2D(None, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn10   = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias10 = L.Bias(shape=(256,)),
            conv11 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn11   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias11 = L.Bias(shape=(512,)),
            conv12 = L.Convolution2D(None, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn12   = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias12 = L.Bias(shape=(256,)),
            conv13 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn13   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias13 = L.Bias(shape=(512,)),

            conv14 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn14   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias14 = L.Bias(shape=(1024,)),
            conv15 = L.Convolution2D(None, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn15   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias15 = L.Bias(shape=(512,)),
            conv16 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn16   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias16 = L.Bias(shape=(1024,)),
            conv17 = L.Convolution2D(None, 512,  ksize=1, stride=1, pad=0, nobias=True),
            bn17   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias17 = L.Bias(shape=(512,)),
            conv18 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn18   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias18 = L.Bias(shape=(1024,)),

            # detection layers
            conv19 = L.Convolution2D(None, 1024,  ksize=3, stride=1, pad=1, nobias=True),
            bn19   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias19 = L.Bias(shape=(1024,)),
            conv20 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn20   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias20 = L.Bias(shape=(1024,)),
            conv21 = L.Convolution2D(None, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn21   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias21 = L.Bias(shape=(1024,)),
            
            conv22 = L.Convolution2D(None, N_BOXES * (5+N_CLASSES), ksize=1, stride=1, pad=0, nobias=True),
            bias22 = L.Bias(shape=(N_BOXES* (5+N_CLASSES),)),
        )
        self.gpu = -1
        if gpu >= 0:
            self.gpu = gpu
            self.to_gpu()
        self.train = False

    def forward(self, x):
        batch_size = x.data.shape[0]

        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias9(self.bn9(self.conv9(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias10(self.bn10(self.conv10(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias11(self.bn11(self.conv11(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias12(self.bn12(self.conv12(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias13(self.bn13(self.conv13(h), test=not self.train)), slope=0.1)
        # TODO: high resolution classifier
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        
        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h), test=not self.train)), slope=0.1)

        h = F.leaky_relu(self.bias19(self.bn19(self.conv19(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias20(self.bn20(self.conv20(h), test=not self.train)), slope=0.1)
        # TODO: high resolution classifier
        h = F.leaky_relu(self.bias21(self.bn21(self.conv21(h), test=not self.train)), slope=0.1)

        h = self.bias22(self.conv22(h))
        return h

    def __call__(self, h, ground_truths):
        batch_size = h.data.shape[0]

        # 推論を実行し結果を抽出
        h = self.forward(h)
        h = F.reshape(h, (batch_size, N_BOXES, 5+N_CLASSES, N_GRID, N_GRID))
        px, py, pw, ph, pconf, pprob \
            = F.split_axis(h, indices_or_sections=(1,2,3,4,5), axis=2)
        pconf = F.sigmoid(pconf)

        # 教師データを初期化
        tx = np.tile(0.5, px.shape) # 基本は0.5 (グリッド中心)
        ty = np.tile(0.5, py.shape)
        tw = np.zeros(ph.shape) # 基本は0 (e^t,e^h = 1となるように)
        th = np.zeros(ph.shape)
        tconf = np.zeros(pconf.shape) # 基本は0
        tprob = pprob.data.copy() # 真のグリッド以外は損失誤差が発生しないよう推定値をコピー

        # scaling factorを初期化
        box_scale_factor = np.tile(SCALE_FACTORS['nocoord'], tconf.shape).astype(np.float32)
        conf_scale_factor = np.tile(SCALE_FACTORS['noconf'], tconf.shape).astype(np.float32)

        # 一定以上のIOUを持つanchorに対する教師データのconfidence scoreは下げない
        best_ious = []
        for batch in range(0, batch_size):
            ious = []
            pboxes = self.all_pred_boxes(px.data[batch], py.data[batch], pw.data[batch], ph.data[batch])
            for truth_box in ground_truths[batch]:
                tboxes = Box(
                    x=np.broadcast_to(np.array(truth_box.left).astype(np.float32), pboxes.left.shape),
                    y=np.broadcast_to(np.array(truth_box.top).astype(np.float32), pboxes.top.shape),
                    width=np.broadcast_to(np.array(truth_box.width).astype(np.float32), pboxes.width.shape),
                    height=np.broadcast_to(np.array(truth_box.height).astype(np.float32), pboxes.height.shape)
                )
                ious.append(Box.iou(pboxes, tboxes))
            best_ious.append(np.asarray(ious))

        best_ious = np.asarray(best_ious).reshape(batch_size, N_BOXES, 1, N_GRID, N_GRID)
        tconf[best_ious > IOU_THRESH] = pconf.data.copy()[best_ious > IOU_THRESH]
        conf_scale_factor[best_ious > IOU_THRESH] = 0

        # objectに最も近いanchor boxに対する教師データをground truthに近づける
        for batch in range(0, batch_size):
            for truth_box in ground_truths[batch]:
                truth_index = 0
                best_iou = 0.
                for anchor_index, anchor_box in enumerate(ANCHOR_BOXES, 0):
                    iou = Box.iou(Box(0., 0., truth_box.width, truth_box.height),
                                  Box(0., 0., anchor_box[0], anchor_box[1]))
                    if best_iou < iou:
                        best_iou = iou
                        truth_index = anchor_index

                grid_x = int(math.modf(truth_box.center.x)[1])
                grid_y = int(math.modf(truth_box.center.y)[1])
                tx[batch, truth_index, :, grid_y, grid_x] = math.modf(truth_box.center.x)[0]
                ty[batch, truth_index, :, grid_y, grid_x] = math.modf(truth_box.center.y)[0]
                tw[batch, truth_index, :, grid_y, grid_x] \
                    = np.log(truth_box.width / ANCHOR_BOXES[truth_index][0])
                th[batch, truth_index, :, grid_y, grid_x] \
                    = np.log(truth_box.height / ANCHOR_BOXES[truth_index][1])
                tprob[batch, truth_index, :, grid_y, grid_x] = 0.
                tprob[batch, truth_index, int(truth_box.clazz), grid_y, grid_x] = 1.
                box_scale_factor[batch, truth_index, :, grid_y, grid_x] = SCALE_FACTORS['coord']
                
                pred_box = Box(
                    x=px.data[batch, truth_index, 0, grid_y, grid_x] + grid_x,
                    y=py.data[batch, truth_index, 0, grid_y, grid_x] + grid_y,
                    width=(np.exp(pw.data[batch, truth_index, 0, grid_y, grid_x]) \
                           * ANCHOR_BOXES[truth_index][0]),
                    height=(np.exp(ph.data[batch, truth_index, 0, grid_y, grid_x]) \
                           * ANCHOR_BOXES[truth_index][1])
                )
                pred_iou = Box.iou(pred_box, truth_box)
                tconf[batch, truth_index, :, grid_y, grid_x] = pred_iou
                conf_scale_factor[batch, truth_index, :, grid_y, grid_x] = SCALE_FACTORS['conf']

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
        batch_size = x.shape[0]
        h = self.forward(x)
        h = F.reshape(h, (batch_size, N_BOXES, 5+N_CLASSES, N_GRID, N_GRID))
        return self.from_variable(h)

    def all_pred_boxes(self, px, py, pw, ph):
        x_offsets = np.broadcast_to(np.arange(N_GRID).astype(np.float32), px.shape)
        y_offsets = np.broadcast_to(np.arange(N_GRID).astype(np.float32), py.shape)
        w_anchors = np.broadcast_to(np.reshape(np.array(ANCHOR_BOXES).astype(np.float32)[:,0], (N_BOXES,1,1,1)), pw.shape)
        h_anchors = np.broadcast_to(np.reshape(np.array(ANCHOR_BOXES).astype(np.float32)[:,0], (N_BOXES,1,1,1)), ph.shape)
        return Box(x=x_offsets + F.sigmoid(px).data, y=y_offsets + F.sigmoid(py).data,
                   width=np.exp(pw) * w_anchors, height=np.exp(ph) * h_anchors)

    def from_variable(self, v):
        return chainer.cuda.to_cpu(v.data)

    def to_variable(self, v):
        v = v.astype(np.float32)
        if self.gpu >= 0:
            v = chainer.cuda.to_gpu(v)
        return chainer.Variable(v)
