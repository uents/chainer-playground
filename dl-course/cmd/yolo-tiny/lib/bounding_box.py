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


class Box():
    def __init__(self, x, y, width, height, clazz=0, objectness=1.0):
        self.left = x
        self.top = y
        self.right = x + width
        self.bottom = y + height
        self.clazz = clazz
        self.objectness = objectness

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def vertex(self):
        return ((self.left, self.top), (self.right-1, self.top),
                (self.left, self.bottom-1), (self.right-1, self.bottom-1))

    def area(self):
        return float(self.width * self.height)

    @classmethod
    def overlap(self, box1, box2):
        left = max(box1.left, box2.left)
        top = max(box1.top, box2.top)
        right = min(box1.right, box2.right)
        bottom = min(box1.bottom, box2.bottom)
        width = max(0, right - left)
        height = max(0, bottom - top)
        return Box(left, top, width, height)

    @classmethod
    def intersection(self, box1, box2):
        return Box.overlap(box1, box2).area()

    @classmethod
    def union(self, box1, box2):
        return box1.area() + box2.area() - Box.intersection(box1, box2)

    @classmethod
    def iou(self, box1, box2):
        return Box.intersection(box1, box2) / Box.union(box1, box2)


class GroundTruth():
    def __init__(self, width, height, bounding_boxes=[]):
        self.width = width
        self.height = height
        self.bounding_boxes = bounding_boxes
