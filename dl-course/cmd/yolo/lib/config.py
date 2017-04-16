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

N_BOXES = 7
ANCHOR_BOXES = np.array([[5.37510, 5.03125],
                         [5.40625, 4.68750],
                         [2.96875, 2.53125],
#                         [2.59375, 2.78125],
                         [2.59375, 7.88282],
                         [6.33843, 2.78125],
                         [1.93754, 3.25843],
                         [0.67738, 0.57273]])

# training configurations
MOMENTUM = 0.9
WEIGHT_DECAY = 0.005
LR_SCHEDULES = {
    '1': 1e-6,
    '101': 1e-5,
}

DROPOUT_RATIO = 0.5
SCALE_FACTORS = {
    'coord': 1.0,
    'nocoord': 0.1,
    'conf': 3.0,
    'noconf': 0.1,
}

# detection configurations
CONFIDENCE_KEEP_THRESH = 0.6
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
