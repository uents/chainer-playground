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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return '<Point x:%4.1f y:%4.1f>' % (self.x, self.y)

class Box():
    def __init__(self, x, y, width, height,
                 confidence=0., clazz=0, objectness=1.):
        self.left = x
        self.top = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.clazz = clazz
        self.objectness = objectness

    def __repr__(self):
        return '<Box x:%4.1f y:%4.1f w:%4.1f h:%4.1f c:%2d o:%1.3f>' % \
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
        if pred_box.clazz != best_truth.clazz:
            return False
        return True


# [real_width, real_height] => [448., 448.]
# 矩形の始点は矩形左上
def real_to_yolo_coord(box, width, height):
    x = box.left * INPUT_SIZE / width
    y = box.top * INPUT_SIZE / height
    w = box.width * INPUT_SIZE / width
    h = box.height * INPUT_SIZE / height
    return Box(x=x, y=y, width=w, height=h,
                confidence=box.confidence,
                clazz=box.clazz,
                objectness=box.objectness)

# [448., 448.] => [7., 7.]
# 矩形の始点は矩形中心
def yolo_to_grid_coord(box):
    grid_size = INPUT_SIZE / N_GRID
    x = (box.left + (box.width / 2.)) / grid_size
    y = (box.top + (box.height / 2.)) / grid_size
    w = box.width / INPUT_SIZE
    h = box.height / INPUT_SIZE
    return Box(x=x, y=y, width=w, height=h,
                confidence=box.confidence,
                clazz=box.clazz,
                objectness=box.objectness)

# [7., 7.] => [448., 448.]
# 矩形の始点は矩形左上
def grid_to_yolo_coord(box, grid_cell):
    grid_size = INPUT_SIZE / N_GRID
    w = box.width * INPUT_SIZE
    h = box.height * INPUT_SIZE
    x = max(((grid_cell.x + box.left) * grid_size) - w / 2., 0.)
    y = max(((grid_cell.y + box.top) * grid_size) - h / 2., 0.)
    w = min(w, INPUT_SIZE - x)
    h = min(h, INPUT_SIZE - y)
    return Box(x=x, y=y, width=w, height=h,
                confidence=box.confidence,
                clazz=box.clazz,
                objectness=box.objectness)

# [448., 448.] => [real_width, real_height]
# 矩形の始点は矩形左上
def yolo_to_real_coord(box, width, height):
    x = box.left * width / INPUT_SIZE
    y = box.top * height / INPUT_SIZE
    w = box.width * width / INPUT_SIZE
    h = box.height * height / INPUT_SIZE
    return Box(x=x, y=y, width=w, height=h,
                confidence=box.confidence,
                clazz=box.clazz,
                objectness=box.objectness)

# YOLO座標系のBox情報をTensor情報に変換
def encode_box_tensor(yolo_box):
    grid_box = yolo_to_grid_coord(yolo_box)
    grid_cell = Point(
        x=int(math.modf(grid_box.left)[1]),
        y=int(math.modf(grid_box.top)[1])
    )
    norm_box = Box(
        x=math.modf(grid_box.left)[0],
        y=math.modf(grid_box.top)[0],
        width=grid_box.width,
        height=grid_box.height,
        confidence=grid_box.confidence,
        clazz=grid_box.clazz,
        objectness=1.0
    )

    tensor = np.zeros(((5*N_BOXES)+N_CLASSES, N_GRID, N_GRID)).astype(np.float32)
    tensor[:5, grid_cell.y, grid_cell.x] \
        = [norm_box.left, norm_box.top, norm_box.width, norm_box.height, 1.0]
    tensor[5:, grid_cell.y, grid_cell.x] \
        = np.eye(N_CLASSES)[np.array(int(grid_box.clazz))] # one hot vector
#    print(tensor)
    return tensor

# Tensor情報をYOLO座標系のBox情報に変換
def decode_box_tensor(tensor):
    px, py, pw, ph, pconf, pprob \
        = np.array_split(tensor, indices_or_sections=(1,2,3,4,5), axis=0)
    px = px.reshape(px.shape[1:])
    py = py.reshape(py.shape[1:])
    pw = pw.reshape(pw.shape[1:])
    ph = ph.reshape(ph.shape[1:])
    pconf = pconf.reshape(pconf.shape[1:])

    # グリッド毎のクラス確率を算出 (N_CLASSES, N_GRID, N_GRID)
    class_prob_map = pprob * pconf
    # 最大クラス確率となるクラスラベルを抽出 (N_GRID, N_GRID)
    class_label_map = class_prob_map.argmax(axis=0)
    # 全てのグリッドマップを算出 (N_GRID, N_GRID)
    grid_map = np.tile(True, pconf.shape)
    # 全てのグリッド位置を算出 (N_GRID, N_GRID)
    grid_cells = [Point(x=float(point[1]), y=float(point[0]))
                    for point in np.argwhere(grid_map)]

    boxes = []
    for i in six.moves.range(0, grid_map.sum()):
        grid_box = Box(x=px[grid_map][i],
                    y=py[grid_map][i],
                    width=pw[grid_map][i],
                    height=ph[grid_map][i],
                    confidence=pconf[grid_map][i],
                    clazz=class_label_map[grid_map][i],
                    objectness=class_prob_map.max(axis=0)[grid_map][i])
        boxes.append(
            {'box': grid_to_yolo_coord(grid_box, grid_cells[i]),
             'grid_cell': grid_cells[i]}
        )
    return boxes

# 検出候補のBounding Boxを選定
def select_candidates(tensor):
    px, py, pw, ph, pconf, pprob \
        = np.array_split(tensor, indices_or_sections=(1,2,3,4,5), axis=0)
    px = px.reshape(px.shape[1:])
    py = py.reshape(py.shape[1:])
    pw = pw.reshape(pw.shape[1:])
    ph = ph.reshape(ph.shape[1:])
    pconf = pconf.reshape(pconf.shape[1:])

    # グリッド毎のクラス確率を算出 (N_CLASSES, N_GRID, N_GRID)
    class_prob_map = pprob * pconf
    # 最大クラス確率となるクラスラベルを抽出 (N_GRID, N_GRID)
    class_label_map = class_prob_map.argmax(axis=0)
    # 最大クラス確率が閾値以上のグリッドを検出候補として抽出 (N_GRID, N_GRID)
    candidate_map = class_prob_map.max(axis=0) > CLASS_PROBABILITY_THRESH
    # 検出候補のグリッド位置を抽出
    grid_cells = [Point(x=float(point[1]), y=float(point[0]))
                    for point in np.argwhere(candidate_map)]

    candidates = []
    for i in six.moves.range(0, candidate_map.sum()):
        grid_box = Box(x=px[candidate_map][i],
                    y=py[candidate_map][i],
                    width=pw[candidate_map][i],
                    height=ph[candidate_map][i],
                    confidence=pconf[candidate_map][i],
                    clazz=class_label_map[candidate_map][i],
                    objectness=class_prob_map.max(axis=0)[candidate_map][i])
        candidates.append(grid_to_yolo_coord(grid_box, grid_cells[i]))
    return candidates

# non-maximal supression
def nms(candidates):
    if len(candidates) == 0:
        return []

    winners = []
    sorted(candidates, key=lambda box: box.objectness, reverse=True)
    winners.append(candidates[0]) # 第１候補は必ず採用

    # 第２候補以降は上位の候補とのIOU次第
    # TODO: 比較すべきは勝ち残ったBoxかも？
    for i, box1 in enumerate(candidates[1:], 1):
        for box2 in candidates[:i]:
            if Box.iou(box1, box2) > IOU_THRESH:
                break
        else:
            winners.append(box1)
    return winners

"""
def final_detection(grid_box, real_width, real_height):
    grid_size = INPUT_SIZE / N_GRID
    box = Box(x=((grid_box.left * grid_size * real_width / INPUT_SIZE) - (grid_box.width * real_width / 2)),
              y=((grid_box.top * grid_size * real_height / INPUT_SIZE) - (grid_box.height * real_height / 2)),
              width=grid_box.width * real_width,
              height=grid_box.height * real_height,
              clazz=grid_box.clazz,
              objectness=grid_box.objectness)
    box.x = max(0., box.left)
    box.y = max(0., box.top)
    box.width = min(box.width, real_width - box.left)
    box.height = min(box.height, real_height - box.top)
    return box

def __detection(self, px, py, pw, ph, pconf, pprob):
    batch_size = px.data.shape[0]
    _px = F.reshape(px, (batch_size, N_GRID, N_GRID)).data
    _py = F.reshape(py, (batch_size, N_GRID, N_GRID)).data
    _pw = F.reshape(pw, (batch_size, N_GRID, N_GRID)).data
    _ph = F.reshape(ph, (batch_size, N_GRID, N_GRID)).data
    _pconf = F.reshape(pconf, (batch_size, N_GRID, N_GRID)).data
    _pprob = pprob.data

    detections = []
    for i in range(0, batch_size):
        candidates = self.__select_candidates(
            _px[i], _py[i], _pw[i], _ph[i], _pconf[i], _pprob[i], self.class_prob_thresh)
        winners = self.__nms(candidates, self.iou_thresh)
        detections.append(winners)
    return detections
"""
