# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import math
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from bounding_box import *


def init_positives():
    return [{'true': 0, 'false': 0}  for i in range(0, N_CLASSES)]

def count_positives(pred_boxes, truth_boxes):
    positives = init_positives()
    for pred_box in pred_boxes:
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
        if tp == 0 and fp == 0: return 0.
        return float(tp) / (tp + fp)
    return [precision(p['true'], p['false']) for p in positives[1:]]

def mean_average_precision(positives):
    aps = average_precisions(positives)
    return np.asarray(aps).mean()

def recall(positives, real_truth_boxes):
    n_positive_truths = np.asarray([p['true'] for p in positives[1:]]).sum()
    n_all_truths = len(real_truth_boxes.ravel())
    return float(n_positive_truths) / n_all_truths
