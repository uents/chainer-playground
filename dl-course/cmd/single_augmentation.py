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
from image_process import *

class Item():
    def __init__(self, item):
        self.color_image_path = item['color_image_path']
        self.label_image_path = item['label_image_path']
        self.bounding_box = Box(
            x=int(item['bounding_boxes'][0]['x']),
            y=int(item['bounding_boxes'][0]['y']),
            w=int(item['bounding_boxes'][0]['width']),
            h=int(item['bounding_boxes'][0]['height'])
        )

def load_catalog(catalog_file):
    try:
        with open(os.path.join(catalog_file), 'r') as fp:
            catalog = json.load(fp)
    except IOError:
        return []

    dataset = filter(lambda item: item['bounding_boxes'] != [], catalog['dataset'])
    return dataset

def find_bitmap(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext == '.bmp':
                yield os.path.join(root, f)

def single_augmentation(args):
    dataset = load_catalog(args.catalog_file)
    items = [Item(item) for item in dataset]
    bg_image_paths = [path for path in find_bitmap(args.bg_image_dir)]

    for count in six.moves.range(1, args.number+1):
        item = np.random.choice(items)
        sys.stdout.write('\r%d process %s' % (count, item.color_image_path))

        obj_image = extract_object_image(
            item.color_image_path, item.label_image_path, item.bounding_box)
        oh, ow = obj_image.shape[:2]

        bg_image_path = np.random.choice(bg_image_paths)
        bg_image = cv2.imread(bg_image_path)
        bh, bw = bg_image.shape[:2]

        corner_x = (bw - ow) / 2 # TODO: 乱数
        corner_y = bh - oh - 128 # TODO: 乱数
        new_image = overlay_image(obj_image, bg_image, corner=(corner_x, corner_y))
        cv2.imwrite('foo.bmp', new_image)
        break
    sys.stdout.write('\n')

def parse_arguments():
    description = 'data augmentation for single images'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--catalog-file', type=str, dest='catalog_file', default='', required=True)
    parser.add_argument('--bg-image-dir', type=str, dest='bg_image_dir', default='', required=True)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default='')
    parser.add_argument('--number', '-n', type=str, dest='number', default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    single_augmentation(args)
    print('done')
