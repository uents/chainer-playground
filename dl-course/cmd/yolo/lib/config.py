# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import numpy as np

# network configurations
N_CLASSES = 26  # 0..25
                # F.softmax_cross_entropy()で扱うラベルが
                # 0始まりの必要があるため、便宜的に0を追加

N_GRID = 7
INPUT_SIZE = N_GRID * 32

N_BOXES = 5
ANCHOR_BOXES = np.array([[5.375, 5.03125],
                         [5.40625, 4.6875],
                         [2.96875, 2.53125],
                         [2.59375, 2.78125],
                         [1.9375, 3.25]])

# training configurations
MOMENTUM = 0.9
WEIGHT_DECAY = 0.05
LR_SCHEDULES = {
    '1' : 1e-6,
    '501' : 1e-5,
}

DROPOUT_RATIO = 0.3
SCALE_FACTORS = {
    'coord': 1.0,
    'nocoord': 0.1,
    'conf': 5.0,
    'noconf': 0.1,
}

# detection configurations
CLASS_PROBABILITY_THRESH = 0.3
IOU_THRESH = 0.3

# file paths
CLASSIFIER_FINAL_MODEL_PATH \
    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '..', 'classifier', 'classifier_final.model')
DETECTOR_FIRST_MODEL_PATH \
    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '..', 'detector', 'detector_first.model')
DETECTOR_FINAL_MODEL_PATH \
    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '..', 'detector', 'detector_final.model')
