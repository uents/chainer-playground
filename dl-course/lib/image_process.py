# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import math
import random
import numpy as np
import cv2
import scipy
import PIL.Image

def extract_object_image(color_image_path, label_image_path, bounding_box):
    color_image = cv2.imread(color_image_path)
    label_image = cv2.imread(label_image_path, 0)

    masked_image = cv2.bitwise_and(color_image, color_image, mask=label_image)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
    masked_image[label_image == 0, 3] = 0

    obj_image = masked_image[\
                    bounding_box.top:bounding_box.bottom,\
                    bounding_box.left:bounding_box.right]
    return obj_image

def scale_image(image, scale):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w*scale), int(h*scale)))

def rotate_image(image, angle):
    return scipy.ndimage.rotate(image, angle, reshape=True)

def overlay_image(obj_image, bg_image, corner=None):
    obj_pimage = PIL.Image.fromarray(cv2.cvtColor(obj_image, cv2.COLOR_BGRA2RGBA))
    new_pimage = PIL.Image.fromarray(cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGBA))
    new_pimage.paste(obj_pimage, box=corner, mask=obj_pimage)
    return cv2.cvtColor(np.asarray(new_pimage), cv2.COLOR_RGBA2BGR)
