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
GRID_SIZE = 32

N_GRID = 9
INPUT_SIZE = GRID_SIZE * N_GRID

ANCHOR_BOXES = np.array([[0.32688071, 0.74313729],
                         [0.64044543, 0.79116386],
                         [0.22818257, 0.40743414],
                         [0.79330671, 0.30941929],
                         [0.49256300, 0.32012029]])
ANCHOR_BOXES *= N_GRID
N_BOXES = int(len(ANCHOR_BOXES))


# training configurations
MOMENTUM = 0.9
WEIGHT_DECAY = 0.005
DROPOUT_RATIO = 0.5 # unused

LR_SCHEDULES = {
    '1': 1e-6,
    '101': 1e-5,
    '4001': 3e-6,
    '7001': 1e-6,
}

SCALE_FACTORS = {
    'coord': 1.0,
    'nocoord': 0.1,
    'conf': 3.0,
    'noconf': 0.1,
}
CONFIDENCE_KEEP_THRESH = 0.6

# inference configurations
CLASS_PROBABILITY_THRESH = 0.3
NMS_IOU_THRESH = 0.3

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
