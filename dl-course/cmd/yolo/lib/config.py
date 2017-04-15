# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os

# network configurations
INPUT_SIZE = 224
N_BOXES = 5
N_GRID = 7
N_CLASSES = 26  # 0..25
                # F.softmax_cross_entropy()で扱うラベルが
                # 0始まりの必要があるため、便宜的に0を追加

# training configurations
MOMENTUM = 0.9
WEIGHT_DECAY = 0.05
LR_SCHEDULES = {
    '1' : 1e-5,
    '3001' : 1e-5,
#    '1' : 1e-7,
#    '101' : 1e-6,
#    '301' : 1e-4,
#    '301' : 1e-5,
#    '501' : 1e-4,
#    '5001' : 3e-5,
#    '10001' : 1e-5,
}

DROPOUT_RATIO = 0.3
SCALE_FACTORS = {
    'coord': 5.0,
    'nocoord': 0.1,
    'obj': 1.0,
    'noobj': 0.1,
    'prob': 1.0
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
