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


def make_result_dict(predicted_boxes, real_width, real_height):
    def to_box_dict(pred_box):
        box = yolo_to_real_coord(pred_box['box'], real_width, real_height)
        return {
            'box': {
                'x': str(box.left),
                'y': str(box.top),
                'width': str(box.width),
                'height': str(box.height),
                'confidence': str(box.confidence),
                'class': str(int(box.clazz)),
                'objectness': str(box.objectness)
            },
            'grid_cell': {
                'x': str(int(pred_box['grid_cell'].x)),
                'y': str(int(pred_box['grid_cell'].y))
            }
        }
    return [to_box_dict(pred_box) for pred_box in predicted_boxes]

def predict(args):
    print('predict: gpu:%d' % (args.gpu))

    model = YoloDetector(args.gpu)
    if len(args.model_file) > 0:
        print('load model: %s' % (args.model_file))
        chainer.serializers.load_npz(args.model_file, model)

    image_paths = load_catalog(args.catalog_file)
    n_test = len(image_paths)
    print('number of dataset: %d' % n_test)

    results = []
    for count, image_path in enumerate(image_paths, 1):
        sys.stdout.write('\r%d predict: %s' % (count, image_path))
        image = Image(image_path, INPUT_SIZE, INPUT_SIZE)
        xs = chainer.Variable(
            xp.asarray(image.image[np.newaxis,:]).transpose(0,3,1,2).astype(np.float32) / 255.)

        model.train = False
        h = model.predict(xs)
        predicted_boxes = decode_box_tensor(h[0])
        results.append({
            'color_image_path': image_path,
            'bounding_boxes': make_result_dict(predicted_boxes, image.real_width, image.real_height)
        })

    with open(args.result_file, 'w') as fp:
        json.dump(results, fp, sort_keys=True, ensure_ascii=False, indent=2)


def parse_arguments():
    description = 'YOLO Detection Predictor'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-file', type=str, dest='model_file', default=DETECTOR_FINAL_MODEL_PATH)
    parser.add_argument('--catalog-file', type=str, dest='catalog_file', default='')
    parser.add_argument('--result-file', type=str, dest='result_file', default='', required=True)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy

    predict(args)
    print('done')
