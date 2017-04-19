# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from yolo_v2 import *
from bounding_box import *
from image_process import *


class YoloPredictor(chainer.Chain):
    def __init__(self, gpu=-1, model_file='',
                 n_grid=N_GRID, anchor_boxes=np.zeros((N_GRID,2))):
        self.n_grid = n_grid
        self.input_size = n_grid * GRID_SIZE
        self.anchor_boxes = anchor_boxes

        self.model = YoloDetector(gpu, n_grid, anchor_boxes)
        if len(model_file) > 0:
            print('load model: %s' % model_file)
            chainer.serializers.load_npz(model_file, self.model)
        self.model.train = False
        self.gpu = gpu

    def predict(self, image_paths):
        xp = chainer.cuda.cupy if self.gpu >= 0 else np
        batch_size = len(image_paths)
        
        # 画像リストをロード
        images = [Image(path, self.input_size) for path in image_paths]
        xs = [image.image for image in images]

        # 推論を実行
        xs = chainer.Variable(xp.asarray(xs).transpose(0,3,1,2).astype(np.float32) / 255.)
        tensors = self.model.predict(xs)

        # 推論結果をBounding Boxに変換
        bounding_boxes = [inference_to_bounding_boxes(tensor, self.anchor_boxes, self.input_size)
                          for tensor in tensors]
        return [[[{'bounding_box': yolo_to_real_coord(bbox['bounding_box'],
                                                      image.real_width, image.real_height,
                                                      self.input_size),
                   'grid_cell': bbox['grid_cell']}
                  for bbox in bboxes_of_anchor]
                 for bboxes_of_anchor in bboxes_of_image]
                for bboxes_of_image, image in zip(bounding_boxes, images)]
