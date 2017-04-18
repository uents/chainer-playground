# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import six
import sys
import os
import math
import random
import numpy as np
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from config import *
from bounding_box import *


def load_catalog(catalog_file):
    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return []
    dataset = filter(lambda item: item['bounding_boxes'] != [], catalog['dataset'])
    return dataset

class Collector():
    def __init__(self, catalog_file=''):
        dataset = load_catalog(catalog_file)
        truth_boxes = np.asarray([[dict_to_box(box) for box in item['bounding_boxes']]
                                  for item in dataset])
        self.mean_ap = 0.
        self.recall = 0.

        # 集計用のデータフレームを用意
        df = pd.DataFrame(columns=['class', 'total', 'true_positive', 'false_positive',
                                   'average_precision', 'average_recall'])
        df['class'] = np.arange(1, N_CLASSES)
        df = df.set_index('class').fillna(0)
        df.ix[:, 'total'] = [len(filter(lambda box: box.clazz == clazz, truth_boxes.ravel()))
                             for clazz in df.index]
        self.df = df

    def validate_bounding_boxes(self, pred_boxes, truth_boxes):
        for pred_box in pred_boxes:
            if int(pred_box.clazz) == 0:
                continue
            correct, iou = Box.correct(pred_box, truth_boxes)
            if correct:
                self.df.ix[int(pred_box.clazz), 'true_positive'] += 1
            else:
                self.df.ix[int(pred_box.clazz), 'false_positive'] += 1

    def update(self):
        self.df.ix[:, 'average_precision'] \
            = self.df.ix[:, 'true_positive'] / (self.df.ix[:, 'true_positive'] + self.df.ix[:, 'false_positive'])
        self.df.ix[:, 'average_recall'] \
            = self.df.ix[:, 'true_positive'] / self.df.ix[:, 'total']
        self.df = self.df.fillna(0)
        self.mean_ap = self.df['average_precision'].mean()
        self.recall = float(self.df['true_positive'].sum()) / self.df['total'].sum()

    def dump(self, dir_path):
        csv_path = os.path.join(dir_path, 'valid.csv')
        with open(csv_path, 'w') as fp:
            self.df.to_csv(fp, encoding='cp932', index=True)
        json_path = os.path.join(dir_path, 'valid.json')
        with open(json_path, 'w') as fp:
            json.dump({'map': self.mean_ap, 'recall': self.recall}, fp,
                      sort_keys=True, ensure_ascii=False, indent=2)
