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
import pandas as pd
import cv2
import json
import jsonschema

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from box import *

def load_catalog(catalog_file):
    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return []
    return catalog['dataset']

def extract_bounding_boxe(label_image, width, height):
    # ラベルの矩形領域と重心位置を取得
    # XXX: このロジックは複数ラベルに対応できない
    _, _, stats, centroids = cv2.connectedComponentsWithStats(label_image)
    for stat, centroid in zip(stats, centroids):
        # 上下左右に少しパディングをつける
        x = max(stat[0]-4, 0)
        y = max(stat[1]-4, 0)
        w = min(stat[2]+24, width)
        h = min(stat[3]+24, height)
        # 面積が極端に小さい・大きい領域は誤検出のため除外
        if w * h <= (width * 2) or w * h >= (width * height  * 0.95):
            continue
        return Box(x=x, y=y, w=w, h=h)
    return None

def make_bg_image(color_image, box):
    color_image[box.top:box.bottom, box.left:box.right]\
        = color_image[box.top:box.bottom, box.left+box.width:box.right+box.width]
#    color_image = cv2.GaussianBlur(color_image, (9,9), 0)
    return color_image

def make_bg_images(args):
    dataset = load_catalog(args.catalog_file)

    bg_images = []
    for item in dataset:
        label_image = cv2.imread(item['label_image_path'], 0)
        height, width = label_image.shape[:2]
        box = extract_bounding_boxe(label_image, width, height)
        if box is None:
            continue
        if ((360 * 270) < box.area or box.aspect < 0.6 or 1.5 < box.aspect):
            continue
        if (box.right + box.width > height):
            continue
        print('process %s' % item['color_image_path'])
        color_image = cv2.imread(item['color_image_path'])
        bg_images.append(make_bg_image(color_image, box))

    for i, bg_image in enumerate(bg_images, 1):
        path = os.path.join(args.output_dir, ('bg_image%02d.bmp' % i))
        print('save %s' % path)
        cv2.imwrite(path, bg_image)

def parse_arguments():
    description = 'background image maker'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--catalog-file', type=str, dest='catalog_file', default='', required=True)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default='', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    make_bg_images(args)
    print('done')
