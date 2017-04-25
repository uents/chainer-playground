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
import chainer.functions as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return '<Point x:{} y:{}>'.format(self.x, self.y)

class Box():
    def __init__(self, x=0., y=0., width=0., height=0., clazz=0, objectness=1.):
        self.left = x
        self.top = y
        self.width = width
        self.height = height
        self.clazz = clazz
        self.objectness = objectness

    def __repr__(self):
        return '<Box x:{} y:{} w:{} h:{} c:{} o:{}>'.format(
            self.left, self.top, self.width, self.height, int(self.clazz), self.objectness)

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def center(self):
        return Point(x=self.left + self.width / 2.,
                     y=self.top + self.height / 2.)

    def area(self):
        return self.width * self.height

    @classmethod
    def overlap(self, box1, box2):
        if type(box1.left) == np.ndarray:
            left = F.maximum(box1.left, box2.left).data
            top = F.maximum(box1.top, box2.top).data
            right = F.minimum(box1.right, box2.right).data
            bottom = F.minimum(box1.bottom, box2.bottom).data
            width = right - left
            width = F.maximum(np.zeros(width.shape).astype(np.float32), width).data
            height = bottom - top
            height = F.maximum(np.zeros(height.shape).astype(np.float32), height).data
        else:
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
        return safe_divide(Box.intersection(box1, box2), Box.union(box1, box2))

    @classmethod
    def best_iou(self, pred_box, truth_boxes):
        ious = np.asarray([Box.iou(pred_box, truth_box) for truth_box in truth_boxes])
        return ious.max(), truth_boxes[ious.argmax()]

    @classmethod
    def correct(self, pred_box, truth_boxes):
        best_score, best_truth = Box.best_iou(pred_box, truth_boxes)
        if best_score <= 0.5:
            return False, best_score
        if pred_box.clazz != best_truth.clazz:
            return False, best_score
        return True, best_score

    
def dict_to_box(box):
    return Box(x=float(box['x']), y=float(box['y']),
            width=float(box['width']), height=float(box['height']),
            clazz=int(box['class']), objectness=1.)

def safe_divide(dividend, divisor):
    if type(divisor) == np.ndarray:
        divisor[divisor < 1e-12] = 1e-12
    else:
        divisor = 1e-12 if divisor < 1e-12 else divisor
    return dividend / divisor

# [real_width, real_height] => [input_size, input_size]
def real_to_yolo_coord(box, width, height, input_size=INPUT_SIZE):
    x = box.left * input_size / width
    y = box.top * input_size / height
    w = box.width * input_size / width
    h = box.height * input_size / height
    return Box(x=x, y=y, width=w, height=h,
               clazz=box.clazz, objectness=box.objectness)

# [input_size, input_size] => [grid_size, grid_size]
def yolo_to_grid_coord(box):
    x = box.left / GRID_SIZE
    y = box.top / GRID_SIZE
    w = box.width / GRID_SIZE
    h = box.height / GRID_SIZE
    return Box(x=x, y=y, width=w, height=h,
               clazz=box.clazz, objectness=box.objectness)

# [grid_size, grid_size] => [input_size, input_size]
def grid_to_yolo_coord(box):
    x = box.left * GRID_SIZE
    y = box.top * GRID_SIZE
    w = box.width * GRID_SIZE
    h = box.height * GRID_SIZE
    return Box(x=x, y=y, width=w, height=h,
               clazz=box.clazz, objectness=box.objectness)

# [input_size, input_size] => [real_width, real_height]
def yolo_to_real_coord(box, width, height, input_size=INPUT_SIZE):
    x = box.left * width / input_size
    y = box.top * height / input_size
    w = box.width * width / input_size
    h = box.height * height / input_size
    return Box(x=x, y=y, width=w, height=h,
               clazz=box.clazz, objectness=box.objectness)

# 推論結果をBounding Boxに変換
def inference_to_bounding_boxes(tensors, anchor_boxes=ANCHOR_BOXES, input_size=INPUT_SIZE):
    def bboxes_of_anchor(tensor, anchor_box):
        px, py, pw, ph, pconf, pprob \
            = np.array_split(tensor, indices_or_sections=(1,2,3,4,5), axis=0)
        px = F.sigmoid(px.reshape(px.shape[1:])).data
        py = F.sigmoid(py.reshape(py.shape[1:])).data
        pw = pw.reshape(pw.shape[1:])
        ph = ph.reshape(ph.shape[1:])
        pconf = F.sigmoid(pconf.reshape(pconf.shape[1:])).data
        pprob = F.sigmoid(pprob).data
        
        # グリッド毎のクラス確率を算出 (N_CLASSES, N_GRID, N_GRID)
        objectness_map = pprob * pconf

        # 最大クラス確率となるクラスラベルを抽出 (N_GRID, N_GRID)
        class_label_map = objectness_map.argmax(axis=0)

        # 全てのグリッドマップと位置を算出 (N_GRID, N_GRID)
        grid_map = np.tile(True, pconf.shape)
        grid_cells = [Point(x=float(p[1]), y=float(p[0])) for p in np.argwhere(grid_map)]

        bboxes = []
        for i in six.moves.range(0, grid_map.sum()):
            if anchor_box[0] == -1. and anchor_box[1] == -1.:
                bw = pw[grid_map][i] * N_GRID
                bh = ph[grid_map][i] * N_GRID
            else:
                bw = np.exp(pw[grid_map][i]) * anchor_box[0]
                bh = np.exp(ph[grid_map][i]) * anchor_box[1]

            bx = max(grid_cells[i].x + px[grid_map][i] - bw/2., 0.)
            by = max(grid_cells[i].y + py[grid_map][i] - bh/2., 0.)
            bw = min(bw, N_GRID - bx)
            bh = min(bh, N_GRID - by)
            bbox = Box(x=bx, y=by, width=bw, height=bh,
                       clazz=class_label_map[grid_map][i],
                       objectness=objectness_map.max(axis=0)[grid_map][i])
            bboxes.append({'bounding_box': grid_to_yolo_coord(bbox),
                           'grid_cell': grid_cells[i]})
        return bboxes

    return [bboxes_of_anchor(tensor, anchor_box)
            for tensor, anchor_box in zip(tensors, anchor_boxes)]


# 検出候補のBounding Boxを選定
def select_candidates(bounding_boxes, thresh=0.):
    candidates = [[bbox['bounding_box'] for bbox in bboxes_of_anchor]
                  for bboxes_of_anchor in bounding_boxes]
    candidates = reduce(lambda x, y: x + y, candidates)
    return filter(lambda c: c.objectness >= thresh, candidates)


# non-maximum suppression
def nms(candidates, thresh=0.):
    if len(candidates) == 0: return []

    winners = []
    sorted(candidates, key=lambda box: box.objectness, reverse=True)
    winners.append(candidates[0]) # 第１候補は必ず採用

    # 第２候補以降は上位の候補とのIOU次第
    # TODO: 比較すべきは勝ち残ったBoxかも？
    for i, box1 in enumerate(candidates[1:], 1):
        for box2 in candidates[:i]:
            if Box.iou(box1, box2) >= thresh:
                break
        else:
            winners.append(box1)
    return winners
