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
import datetime as dt
import pprint
import numpy as np
import cv2
import json

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from yolo_predictor import *
from bounding_box import *
from image_process import *
from collector import *

xp = np
pp = pprint.PrettyPrinter(indent=2)

START_TIME = dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'valid_' + START_TIME)

def load_catalog(catalog_file):
    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return []
    dataset = filter(lambda item: item['bounding_boxes'] != [], catalog['dataset'])
    return dataset

def validate(args):
    print('validate: gpu:%d class_prob_thresh:%1.2f nms_iou_thresh:%1.2f' %
          (args.gpu, args.class_prob_thresh, args.nms_iou_thresh))
    collector = Collector(args.catalog_file)

    # 推論モデルをロード
    model = YoloPredictor(args.gpu, args.model_file)

    # データセットの読み出し
    dataset = load_catalog(args.catalog_file)
    image_paths = np.asarray([item['color_image_path'] for item in dataset])
    real_truth_boxes = np.asarray([[dict_to_box(box) for box in item['bounding_boxes']]
                              for item in dataset])
    n_valid = len(dataset)
    print('number of dataset: %d' % n_valid)
    if n_valid == 0: return

    for count in six.moves.range(0, n_valid, 10):
        # 推論を実行
        ix = np.arange(count, min(count+10, n_valid))
        truth_boxes = real_truth_boxes[ix]
        bounding_boxes = model.predict(image_paths[ix])

        # バウンディングボックスの絞り込み
        for batch in six.moves.range(0, len(ix)):
            candidates = select_candidates(bounding_boxes[batch], args.class_prob_thresh)
            winners = nms(candidates, args.nms_iou_thresh)
            collector.validate_bounding_boxes(winners, truth_boxes[batch])
            for winner, truth_box in itertools.product(winners, truth_boxes[batch]):
                correct, iou = Box.correct(winner, [truth_box])
                print('{0} {1} {2:.3f} pred:{3} truth:{4}'.format(
                    count + batch + 1, correct, iou, winner, truth_box))

    collector.update()
    print('map:%f recall:%f' % (collector.mean_ap, collector.recall))
    collector.dump(SAVE_DIR)

def parse_arguments():
    description = 'YOLO Detection Predictor'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-file', type=str, dest='model_file', default='')
    parser.add_argument('--catalog-file', type=str, dest='catalog_file', default='', required=True)
    parser.add_argument('--class-prob-thresh', type=float, dest='class_prob_thresh', default=0.3)
    parser.add_argument('--nms-iou-thresh', type=float, dest='nms_iou_thresh', default=0.3)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    return parser.parse_args()

def save_params(args):
    params = {
        'elapsed_time': {
            'start': START_TIME,
            'end': dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        },
        'catalog_file': args.catalog_file,
        'model_file': args.model_file,
        'class_prob_thresh': args.class_prob_thresh,
        'nms_iou_thresh': args.nms_iou_thresh
    }
    with open(os.path.join(SAVE_DIR, 'params.json'), 'w') as fp:
        json.dump(params, fp, sort_keys=True, ensure_ascii=False, indent=2)
    
if __name__ == '__main__':
    args = parse_arguments()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    validate(args)
    save_params(args)

    print('done')
