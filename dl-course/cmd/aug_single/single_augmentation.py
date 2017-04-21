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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'lib'))
from box import *
from image_process import *

class Item():
    def __init__(self, item):
        self.color_image_path = item['color_image_path']
        self.label_image_path = item['label_image_path']
        self.clazz = item['classes'][0]
        self.bounding_box = Box(
            x=int(item['bounding_boxes'][0]['x']),
            y=int(item['bounding_boxes'][0]['y']),
            w=int(item['bounding_boxes'][0]['width']),
            h=int(item['bounding_boxes'][0]['height']),
            clazz=int(item['bounding_boxes'][0]['class'])
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

def load_class_table():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'lib', 'item_table.csv')
    with open(path, 'r') as fp:
        df = pd.read_csv(fp, encoding='cp932')
    return df

def find_color_by_class(df, clazz=0):
    return int(df.ix[int(clazz),'r']), \
            int(df.ix[int(clazz),'g']), int(df.ix[int(clazz),'b'])

def single_augmentation(args):
    dataset = load_catalog(args.catalog_file)
    items = [Item(item) for item in dataset]
    bg_image_paths = [path for path in find_bitmap(args.bg_image_dir)]
    df_class_color_table = load_class_table()

    new_dataset = []
    for count in six.moves.range(1, args.number+1):
        # 背景画像を無作為に抽出
        bg_image_path = np.random.choice(bg_image_paths)
        bg_image = cv2.imread(bg_image_path)

        # オブジェクト画像を無作為に抽出
        item = np.random.choice(items)
        sys.stdout.write('\r%d process %s' % (count, item.color_image_path))
        sys.stdout.flush()
        obj_image = extract_object_image(
            item.color_image_path, item.label_image_path, item.bounding_box)

        # オブジェクト画像を回転・拡縮
        if args.with_scaling:
            obj_image = rotate_image(obj_image,
                            random.choice([0, 0, 90, -90, 180]))
        if args.with_scaling:
            obj_image = scale_image(obj_image, random.uniform(0.9, 1.1))

        # オブジェクト画像と背景画像を合成 (合成座標もここで決定)
        new_color_image, bbox = overlay_image(obj_image, bg_image)

        r, g, b = find_color_by_class(df_class_color_table, item.clazz)
        label_image = obj_image.copy()
        label_image[obj_image[:,:,3] > 0] = [b, g, r, 255]

        # ラベル画像と黒画像を合成
        black_image = np.tile(0, bg_image.shape).astype(np.uint8)
        new_label_image, _ = overlay_image(
            label_image, black_image, (bbox.left, bbox.top))

        # バウンディングボックスを画像内に収まるように補正
        h, w = new_color_image.shape[:2]
        bx = max(bbox.left, 0)
        by = max(bbox.top, 0)
        bw = min(bbox.width, bbox.width + bbox.left, w - bbox.left, w)
        bh = min(bbox.height, bbox.height + bbox.top, h - bbox.top, h)

        # 画像サイズを1/2にする
        new_color_image = cv2.resize(new_color_image, (w/2, h/2), cv2.INTER_LINEAR)
        new_label_image = cv2.resize(new_label_image, (w/2, h/2), cv2.INTER_LINEAR)
        bbox = Box(x=bx/2, y=by/2, w=bw/2, h=bh/2, clazz=bbox.clazz)

        # 保存先のディレクトリを作成
        dir_path = os.path.join(args.output_dir, item.clazz.zfill(2))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 画像データを保存
        prefix = item.clazz.zfill(2) + '_' + str(count).zfill(5)
        new_color_image_path = os.path.join(dir_path, prefix + '_color.bmp')
        new_label_image_path = os.path.join(dir_path, prefix + '_label.bmp')
        cv2.imwrite(new_color_image_path, new_color_image)
        cv2.imwrite(new_label_image_path, new_label_image)

        # カタログ情報に追加
        new_dataset.append({
            'classes': [item.clazz],
            'pattern_id': str(count).zfill(5),
            'color_image_path': os.path.abspath(new_color_image_path),
            'label_image_path': os.path.abspath(new_label_image_path),
            'bounding_boxes': [{
                'class': item.clazz, 'x': str(bbox.left), 'y': str(bbox.top),
                'width': str(bbox.width), 'height': str(bbox.height)
            }]
        })

        catalog_path = os.path.join(args.output_dir, 'catalog.json')
        with open(catalog_path, 'w') as fp:
            json.dump({'dataset': new_dataset}, fp,
                      sort_keys=True, ensure_ascii=False, indent=2)

def parse_arguments():
    description = 'data augmentation for single images'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--catalog-file', type=str, dest='catalog_file', default='', required=True)
    parser.add_argument('--bg-image-dir', type=str, dest='bg_image_dir', default='', required=True)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default='Single')
    parser.add_argument('--number', '-n', type=int, dest='number', default=1)
    parser.add_argument('--with-rotation', type=bool, dest='with_rotation', default=True)
    parser.add_argument('--with-scaling', type=bool, dest='with_scaling', default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    single_augmentation(args)
    print('done')
