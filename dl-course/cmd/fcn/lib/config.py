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

#INPUT_SIZE = 224
INPUT_SIZE = 160


# training configurations
MOMENTUM = 0.9
WEIGHT_DECAY = 0.005

LR_SCHEDULES = {
    '1': 1e-4,
    '101': 1e-3,
    '4001': 3e-3,
    '7001': 1e-4,
}
