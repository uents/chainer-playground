# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import math
import random
import numpy as np
import cv2
import scipy.ndimage
import PIL.Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'lib'))
from box import *

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

def random_hsv_image(image, delta_hue=0.01, delta_sat_scale=0.05, delta_val_scale=0.1):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # hue
    hsv_image[:,:,0] += int((np.random.rand() * delta_hue * 2 - delta_hue) * 255)
    # saturation
    sat_scale = 1 + np.random.rand() * delta_sat_scale * 2 - delta_sat_scale
    hsv_image[:,:,1] *= sat_scale
    # value
    val_scale = 1 + np.random.rand() * delta_val_scale * 2 - delta_val_scale
    hsv_image[:,:,2] *= val_scale

    hsv_image[hsv_image < 0] = 0
    hsv_image[hsv_image > 255] = 255
    hsv_image = hsv_image.astype(np.uint8)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def overlay_image(obj_image, bg_image, corner=None):
    oh, ow = obj_image.shape[:2]
    if corner is None:
        bh, bw = bg_image.shape[:2]
        ox = int(random.uniform(0., bw-ow))
        oy = int(random.uniform(0.85*bh - oh, 0.95*bh - oh))
    else:
        ox, oy = corner
    bounding_box = Box(x=ox, y=oy, w=ow, h=oh)

    obj_pimage = PIL.Image.fromarray(cv2.cvtColor(obj_image, cv2.COLOR_BGRA2RGBA))
    new_pimage = PIL.Image.fromarray(cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGBA))
    new_pimage.paste(obj_pimage, box=(ox, oy), mask=obj_pimage)
    return cv2.cvtColor(np.asarray(new_pimage), cv2.COLOR_RGBA2BGR), bounding_box
