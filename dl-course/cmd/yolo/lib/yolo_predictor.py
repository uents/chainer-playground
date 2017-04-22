# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import numpy as np
import json

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from yolo_v2 import *
from bounding_box import *
from image import *


class YoloPredictor(chainer.Chain):
    def __init__(self, gpu=-1, model_file=''):
        if len(model_file) > 0:
            params_file = os.path.join(os.path.split(model_file)[0], 'params.json')
            with open(params_file, 'r') as fp:
                params = json.load(fp)
            self.n_grid = int(params['grid_cells'])
            self.input_size = GRID_SIZE * self.n_grid
            self.anchor_boxes = np.asarray(eval(params['anchor_boxes']))
        else:
            self.n_grid = N_GRID
            self.input_size = INPUT_SIZE
            self.anchor_boxes = ANCHOR_BOXES

        self.model = YoloDetector(gpu, self.n_grid, self.anchor_boxes)
        if len(model_file) > 0:
            print('load model: %s' % model_file)
            chainer.serializers.load_npz(model_file, self.model)
        self.model.train = False
        self.gpu = gpu

    def inference(self, image_paths):
        xp = chainer.cuda.cupy if self.gpu >= 0 else np
        batch_size = len(image_paths)

        # 画像リストをロード
        images = [Image(path, self.input_size) for path in image_paths]
        xs = [image.image for image in images]

        # 推論を実行
        xs = chainer.Variable(xp.asarray(xs).transpose(0,3,1,2).astype(np.float32) / 255.)
        tensors = self.model.predict(xs)
        bounding_boxes = [inference_to_bounding_boxes(tensor, self.anchor_boxes, self.input_size)
                          for tensor in tensors]
        return [[[{'bounding_box': yolo_to_real_coord(bbox['bounding_box'],
                                    image.real_width, image.real_height, self.input_size),
                   'grid_cell': bbox['grid_cell']}
                    for bbox in bboxes_of_anchor]
                    for bboxes_of_anchor in bboxes_of_image]
                    for bboxes_of_image, image in zip(bounding_boxes, images)]

    def final_detection(self, bounding_boxes,
                class_prob_thresh=0.3, nms_iou_thresh=0.3):
        return [nms(select_candidates(bbox_of_image, class_prob_thresh), nms_iou_thresh)
                    for bbox_of_image in bounding_boxes]
