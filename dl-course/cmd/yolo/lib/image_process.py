# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import argparse
import math
import random
import numpy as np
import cv2

class Image():
    def __init__(self, path, width, height):
        self.path = path
        image = cv2.imread(path)
        self.real_height, self.real_width, _ = image.shape
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(image, (width, height))
