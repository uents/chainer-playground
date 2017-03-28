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
        self.width = width
        self.height = height
        self.clazz = clazz
        self.objectness = objectness

    def __repr__(self):
        return "<Box x:%04.1f y:%04.1f w:%04.1f h:%04.1f c:%2d o:%01.3f>" % \
            (self.left, self.top, self.width, self.height, int(self.clazz), self.objectness)

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

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
        union = Box.union(box1, box2)
        return Box.intersection(box1, box2) / Box.union(box1, box2) if union > 0 else 0.0

    @classmethod
    def best_iou(self, pred_box, truth_boxes):
        ious = np.asarray([Box.iou(pred_box, truth_box) for truth_box in truth_boxes])
        return ious.max(), truth_boxes[ious.argmax()]

    @classmethod
    def correct(self, pred_box, truth_boxes):
        best_score, best_truth = Box.best_iou(pred_box, truth_boxes)
        if best_score <= 0.5:
            return False
        elif pred_box.clazz != best_truth.clazz:
            return False
        return True


class GroundTruth():
    def __init__(self, width, height, bounding_boxes=[]):
        self.width = width
        self.height = height
        self.bounding_boxes = bounding_boxes
