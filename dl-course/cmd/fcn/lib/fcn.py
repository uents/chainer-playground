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


class Fcn(chainer.Chain):
    '''
    Fully Convolutional Networks
    '''
    def __init__(self, gpu=-1):
        super(Fcn, self).__init__(
#            conv1  = L.Convolution2D(3,  64, ksize=3, stride=1, pad=1),
            conv1  = L.Convolution2D(3,  64, ksize=3, stride=1, pad=1, nobias=True),
            bn1    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias1  = L.Bias(shape=(64,)),
#            conv2  = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1),
            conv2  = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, nobias=True),
            bn2    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias2  = L.Bias(shape=(64,)),

#            conv3  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1),
            conv3  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn3    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias3  = L.Bias(shape=(128,)),
#            conv4  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1),
            conv4  = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, nobias=True),            
            bn4    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias4  = L.Bias(shape=(128,)),

#            conv5  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1),
            conv5  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn5    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias5  = L.Bias(shape=(256,)),
#            conv6  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1),
            conv6  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, nobias=True),            
            bn6    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias6  = L.Bias(shape=(256,)),
#            conv7  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1),
            conv7  = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, nobias=True),            
            bn7    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias7  = L.Bias(shape=(256,)),
            score_pool3 = L.Convolution2D(None, N_CLASSES, ksize=1, stride=1, pad=0),

#            conv8  = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1),
            conv8  = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),            
            bn8    = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias8  = L.Bias(shape=(512,)),
#            conv9  = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1),
            conv9  = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),            
            bn9    = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias9  = L.Bias(shape=(512,)),
#            conv10 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1),
            conv10 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),            
            bn10   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias10 = L.Bias(shape=(512,)),
            score_pool4 = L.Convolution2D(None, N_CLASSES, ksize=1, stride=1, pad=0),

#            conv11 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1),
            conv11 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),            
            bn11   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias11 = L.Bias(shape=(512,)),
#            conv12 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1),
            conv12 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),            
            bn12   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias12 = L.Bias(shape=(512,)),
#            conv13 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1),
            conv13 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, nobias=True),            
            bn13   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias13 = L.Bias(shape=(512,)),
            score_pool5 = L.Convolution2D(None, N_CLASSES, ksize=1, stride=1, pad=0),

            upsample_32x_pool1 = L.Deconvolution2D(N_CLASSES, N_CLASSES, ksize=64, stride=32, pad=16),
            upsample_2x_pool1  = L.Deconvolution2D(N_CLASSES, N_CLASSES, ksize=4,  stride=2,  pad=1),
            upsample_16x_pool2 = L.Deconvolution2D(N_CLASSES, N_CLASSES, ksize=32, stride=16, pad=8),
            upsample_2x_pool2  = L.Deconvolution2D(N_CLASSES, N_CLASSES, ksize=4,  stride=2,  pad=1),
            upsample_8x_pool3  = L.Deconvolution2D(N_CLASSES, N_CLASSES, ksize=16, stride=8,  pad=4),
        )
        self.gpu = -1
        if gpu >= 0:
            self.gpu = gpu
            self.to_gpu()
        self.train = False

    def forward(self, x):
        # pool1
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        # pool2
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        # pool3
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        p3 = self.score_pool3(h) # (batch_size, N_CLASSES, 28, 28)

        # pool4
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        p4 = self.score_pool4(h) # (batch_size, N_CLASSES, 14, 14)

        # pool5
        h = F.relu(self.conv11(h))
        h = F.relu(self.conv12(h))
        h = F.relu(self.conv13(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        p5 = self.score_pool5(h) # (batch_size, N_CLASSES, 7, 7)

        # FCN-32s
        h32 = self.upsample_32x_pool1(p5)
        u5  = self.upsample_2x_pool1(p5)
        # FCN-16s
        h16 = self.upsample_16x_pool2(u5 + p4)
        u4  = self.upsample_2x_pool2(u5 + p4)
        # FCN-8s
        h8  = self.upsample_8x_pool3(u4 + p3)
        return h8, h16, h32

    def __call__(self, x, t):
        h, _, _ = self.forward(x)
        self.loss = F.softmax_cross_entropy(h, t)
        self.pixel_acc = self.pixel_accuracy(h, t)
        self.h = F.softmax(h)
        return self.loss

    def pixel_accuracy(self, h, t):
        label_mask = t.data != 0
        pred = np.argmax(h.data, axis=1)
        truth = t.data
        return (pred[label_mask] == truth[label_mask]).mean()

    def inference(self, x):
        h, _, _, = self.forward(x)
        h = F.softmax(h)
        return np.argmax(h.data, axis=1)