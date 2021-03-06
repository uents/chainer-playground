# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import math
import random
import numpy as np
import cv2

class Image():
    def __init__(self, path, input_size):
        self.path = path
        image = cv2.imread(path)
        self.real_height, self.real_width, _ = image.shape
        self.image = cv2.resize(image, (input_size, input_size),
                                interpolation=cv2.INTER_LINEAR)
