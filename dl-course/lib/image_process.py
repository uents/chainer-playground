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

def overlay_image(obj_image, bg_image):
    oh, ow = obj_image.shape[:2]
    bh, bw = bg_image.shape[:2]
    ox = int((bw-ow)/2 + random.uniform(-0.4*(bw-ow), 0.4*(bw-ow)))
    oy = int((bh-oh)   - max(random.uniform(0.1*(bh-oh), 0.3*(bh-oh)), 0.05*bh))
    box = Box(x=ow, y=oy, w=ow, h=oh)

    obj_pimage = PIL.Image.fromarray(cv2.cvtColor(obj_image, cv2.COLOR_BGRA2RGBA))
    new_pimage = PIL.Image.fromarray(cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGBA))
    new_pimage.paste(obj_pimage, box=(ox, ow), mask=obj_pimage)
    return cv2.cvtColor(np.asarray(new_pimage), cv2.COLOR_RGBA2BGR), box
