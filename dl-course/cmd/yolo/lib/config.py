# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os

# network configurations
INPUT_SIZE = 448
N_CNN_LAYER = 7
N_BOXES = 1
N_GRID = 7
N_CLASSES = 26  # 0..25
                # F.softmax_cross_entropy()で扱うラベルが
                # 0始まりの必要があるため、便宜的に0を追加

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
