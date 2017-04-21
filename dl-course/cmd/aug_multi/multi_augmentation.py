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

def get_total_width(images):
    return reduce(lambda x, y: x + y, [image.shape[1] for image in images])

def multi_augmentation(args):
    dataset = load_catalog(args.catalog_file)
    all_items = [Item(item) for item in dataset]
    bg_image_paths = [path for path in find_bitmap(args.bg_image_dir)]
    df_class_color_table = load_class_table()

    new_dataset = []
    for count in six.moves.range(1, args.number+1):
        # 背景画像を無作為に抽出
        bg_image_path = np.random.choice(bg_image_paths)
        new_color_image = cv2.imread(bg_image_path)
        bh, bw = new_color_image.shape[:2]

        # ラベル画像用に黒背景を生成
        new_label_image = np.tile(0, new_color_image.shape).astype(np.uint8)

        # データを無作為に抽出
        items = np.random.choice(all_items, size=args.sampling)

        # オブジェクト画像を生成
        obj_images = []
        for item in items:
            sys.stdout.write('\r%d process %s' % (count, item.color_image_path))
            sys.stdout.flush()
            obj_image = extract_object_image(
                item.color_image_path, item.label_image_path, item.bounding_box)
            obj_image = rotate_image(obj_image, random.choice([0, 0, 90, -90, 180]))
            obj_image = scale_image(obj_image, random.uniform(0.9, 1.1))
            obj_images.append(obj_image)

        # 全体に収まるまでオブジェクト数を調整する
        while len(obj_images) > 1 and get_total_width(obj_images) > bw:
            del obj_images[-1]

        # オブジェクトの配置を決定
        bboxes = []
        for i in range(0, len(obj_images)):
            oh, ow = obj_images[i].shape[:2]
            if i == 0:
                if len(obj_images) >= 2:
                    kind = np.random.randint(0,3)
                    if kind == 0:
                        ox = 0
                    elif kind == 1:
                        ox = bw - get_total_width(obj_images)
                    else:
                        ox = bw - get_total_width(obj_images)/2
                else:
                    ox = int(random.uniform(0., bw-ow))
            else:
                ox = bboxes[i-1].right
            oy = int(random.uniform(0.85*bh - oh, 0.95*bh - oh))
            bboxes.append(Box(x=ox, y=oy, w=ow, h=oh, clazz=items[i].clazz))

        new_bboxes = []
        for obj_image, bbox in zip(obj_images, bboxes):
            # オブジェクト画像、ラベル画像を合成
            new_color_image, _ = overlay_image(obj_image, new_color_image,
                                               (bbox.left, bbox.top))

            r, g, b = find_color_by_class(df_class_color_table, bbox.clazz)
            label_image = obj_image.copy()
            label_image[obj_image[:,:,3] > 0] = [b, g, r, 255]
            new_label_image, _ = overlay_image(label_image, new_label_image,
                                               (bbox.left, bbox.top))

            # バウンディングボックスを画像内に収まるように補正
            # (サイズは先に1/2しておく)
            new_bboxes.append(Box(x=max(bbox.left/2, 0),  y=max(bbox.top/2, 0),
                                  w=bbox.width/2, h=bbox.height/2, clazz=bbox.clazz))

        # 画像サイズを1/2にする
        new_color_image = cv2.resize(new_color_image, (bw/2, bh/2), cv2.INTER_LINEAR)
        new_label_image = cv2.resize(new_label_image, (bw/2, bh/2), cv2.INTER_LINEAR)

        # 保存先のディレクトリを作成
        dir_path = os.path.join(args.output_dir, items[0].clazz.zfill(2))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 画像データを保存
        prefix = items[0].clazz.zfill(2) + '_' + str(count).zfill(5)
        new_color_image_path = os.path.join(dir_path, prefix + '_color.bmp')
        new_label_image_path = os.path.join(dir_path, prefix + '_label.bmp')
        cv2.imwrite(new_color_image_path, new_color_image)
        cv2.imwrite(new_label_image_path, new_label_image)

        # カタログ情報に追加
        new_dataset.append({
            'classes': [bbox.clazz for bbox in new_bboxes],
            'pattern_id': str(count).zfill(5),
            'color_image_path': os.path.abspath(new_color_image_path),
            'label_image_path': os.path.abspath(new_label_image_path),
            'bounding_boxes': [{
                'class': bbox.clazz, 'x': str(bbox.left), 'y': str(bbox.top),
                'width': str(bbox.width), 'height': str(bbox.height)
            } for bbox in new_bboxes]
        })

    catalog_path = os.path.join(args.output_dir, 'multi_aug_catalog.json')
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
    parser.add_argument('--sampling', '-s', type=int, dest='sampling', default=4)    
    parser.add_argument('--with-rotation', type=bool, dest='with_rotation', default=True)
    parser.add_argument('--with-scaling', type=bool, dest='with_scaling', default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    multi_augmentation(args)
    print('done')
