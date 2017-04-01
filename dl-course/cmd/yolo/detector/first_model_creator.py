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
import cv2
import json

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from yolo import *


def copy_conv_layer(src, dst):
    for i in range(1, N_CNN_LAYER+1):
        src_layer = eval('src.conv%d' % i)
        dst_layer = eval('dst.conv%d' % i)
        dst_layer.W = src_layer.W
        dst_layer.b = src_layer.b

def parse_arguments():
    description = 'First Model Creator for YOLO Detector'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--input-model-file', type=str, dest='input_model_file', default=CLASSIFIER_FINAL_MODEL_PATH)
    parser.add_argument(
        '--output-model-file', type=str, dest='output_model_file', default=DETECTOR_FIRST_MODEL_PATH)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print('create model: %s => %s' % \
        (args.input_model_file, args.output_model_file))

    classifier_model = YoloClassifier(args.gpu)
    chainer.serializers.load_npz(args.input_model_file, classifier_model)

    '''
    serializers.save_npz()実行時の
    "ValueError: uninitialized parameters cannot be serialized" を回避するために
    ダミーデータでの順伝播を実行する
    '''
    detector_model = YoloDetector(args.gpu)
    dummy_image = chainer.Variable(np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32))
    detector_model.forward(dummy_image)
    copy_conv_layer(classifier_model, detector_model)

    print('save model')
    chainer.serializers.save_npz(args.output_model_file, detector_model)
    print('done')
