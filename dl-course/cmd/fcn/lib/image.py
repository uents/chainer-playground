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

# (batch_size, channels, rows, cols) => (batch_size, rows, cols)
def indexed_label_image(bgr_image):
    b_ix, g_ix, r_ix = bgr_image[:,:,0]/127, bgr_image[:,:,1]/127, bgr_image[:,:,2]/127
    indexed_image = b_ix*9 + g_ix*3 + r_ix
    return indexed_image

# (batch_size, rows, cols) => (batch_size, channels, rows, cols)
def color_label_image(indexed_image):
    b, x = divmod(indexed_image, 9)
    g, r = divmod(x, 3)
    bgr_image = np.asarray([b*128, g*128, r*128])
    bgr_image[bgr_image > 255] = 255
    return bgr_image.astype(np.uint8).transpose(1,2,3,0)
