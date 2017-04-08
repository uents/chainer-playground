# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os

class Box():
    def __init__(self, x, y, w, h, clazz=0):
        self.left = x
        self.top = y
        self.width = w
        self.height = h
        self.clazz = clazz

    def __repr__(self):
        return '<Box x:%4.1f y:%4.1f w:%4.1f h:%4.1f c:%d>' % \
            (self.left, self.top, self.width, self.height, self.clazz)

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def area(self):
        return self.width * self.height

    @property
    def aspect(self):
        return float(self.width) / (self.height)
