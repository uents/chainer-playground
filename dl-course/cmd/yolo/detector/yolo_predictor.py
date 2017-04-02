# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import argparse
import math
import random
import itertools
import numpy as np
import cv2
import json

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from yolo import *
from bounding_box import *
from image_process import *

xp = np


def load_catalog(catalog_file):
    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return []

    dataset = catalog['dataset']
    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    return image_paths

def tensor_to_boxes(tensor):
    return nms(select_candidates(tensor))

def predict(args):
    print('predict: gpu:%d' % (args.gpu))

    model = YoloDetector(args.gpu)
    if len(args.model_file) > 0:
        print('load model: %s' % (args.model_file))
        chainer.serializers.load_npz(args.model_file, model)

    image_paths = load_catalog(args.catalog_file)
    n_test = len(image_paths)
    print('number of dataset: %d' % n_test)

    for count, image_path in enumerate(image_paths, 1):
        image = (Image(image_path, INPUT_SIZE, INPUT_SIZE).image)[np.newaxis,:]
        xs = chainer.Variable(xp.asarray(image).transpose(0,3,1,2).astype(np.float32) / 255.)

        model.train = False
        h = model.forward(xs).data
        predicted_boxes = decode_box_tensor(h[0])
#        print(predicted_boxes)

def parse_arguments():
    description = 'YOLO Detection Predictor'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-file', type=str, dest='model_file', default=DETECTOR_FINAL_MODEL_PATH)
    parser.add_argument('--catalog-file', type=str, dest='catalog_file', default='')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy

    predict(args)
    print('done')
