# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
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

item_table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'lib', 'item_table.csv')

'''
ラベル画像ファイルを探索する
'''
def find_label_images(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f == str('label.bmp'):
                yield os.path.join(root, f)

'''
ラベルの矩形領域とクラス値を抽出する
'''
def extract_bounding_boxes(path, df_items):
    boxes = []
    # ラベルの矩形領域と重心位置を取得
    # XXX:このロジックは複数ラベルに対応できない
    label_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    width, height, _ = label_image.shape
    label_gray = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(label_gray)

    for stat, centroid in zip(stats, centroids):
        b_x, b_y, b_w, b_h, _ = stat[0], stat[1], stat[2], stat[3], stat[4]
        c_x, c_y = int(math.floor(centroid[1])), int(math.floor(centroid[0]))
        # 面積が極端に小さい・大きい領域は誤検出のため除外
        if b_w * b_h <= (width * 2) or b_w * b_h >= (width * height  * 0.95):
            continue
        # 重心位置の画素からクラス値を決定
        r, g, b = label_image[c_x][c_y]
        clazz = int(df_items[(df_items['r'] == r) & (df_items['g'] == g) & (df_items['b'] == b)]['class'].values[0])
        # 結果を格納
        boxes.append({
            'class': str(clazz), 'x': str(b_x), 'y': str(b_y), 'width': str(b_w), 'height': str(b_h)
        })
    return boxes

'''
学習データセットのカタログの各アイテム情報を生成する
'''
def make_catalog_item(camera_image_dir, label_image_dir, label_image_path, df_items):
    _, clazz, pattern_id, _ = label_image_path.split(label_image_dir)[1].split('/')
    classes = clazz.split('_')
    color_image_path = os.path.join(camera_image_dir, clazz, pattern_id, 'color.bmp')
    depth_image_path = os.path.join(camera_image_dir, clazz, pattern_id, 'depth.bmp.bmp')
    bounding_boxes = extract_bounding_boxes(label_image_path, df_items)
    return {
        'classes': classes,
        'pattern_id': pattern_id,
        'color_image_path': os.path.abspath(color_image_path),
        'depth_image_path': os.path.abspath(depth_image_path),
        'label_image_path': os.path.abspath(label_image_path),
        'bounding_boxes': bounding_boxes
    }


'''
学習データセットのカタログ情報を生成する
'''
def make_catalog(input_dir, train_ratio=0.8):
    camera_image_dir = os.path.join(input_dir, 'Single')
    label_image_dir = os.path.join(input_dir, 'mask_label', 'single')

    # クラスとRGB値の関係テーブルをロード
    df_items = pd.read_csv(item_table_path, encoding='cp932')

    # ラベル画像を起点にカタログ情報を収集
    dataset = []
    for count, path in enumerate(find_label_images(label_image_dir), 1):
        sys.stdout.write('\r%d parse %s' % (count, path))
        item = make_catalog_item(camera_image_dir, label_image_dir, path, df_items)
        dataset.append(item)
    sys.stdout.write('\n')

    # カタログ情報を訓練用とクロスバリデーション用に分割
    train_dataset, cv_dataset = [], []
    classes = np.unique(np.asarray([item['classes'] for item in dataset]).ravel())
    for clazz in classes:
        each_dataset = np.asarray(filter(lambda item: item['classes'] == [clazz], dataset))
        all_ixs = range(0, len(each_dataset))
        train_ixs = random.sample(all_ixs, int(len(all_ixs) * train_ratio))
        cv_ixs = list(set(all_ixs) - set(train_ixs))
        train_dataset += each_dataset[train_ixs].tolist()
        cv_dataset += each_dataset[cv_ixs].tolist()

    print('number of dataset: train:%d cv:%d' % (len(train_dataset), len(cv_dataset)))
    return {'dataset': train_dataset}, {'dataset': cv_dataset}

def parse_arguments():
    description = 'single image catalog maker'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input-dir', type=str, dest='input_dir', required=True)
    parser.add_argument('--train-catalog-file', type=str, dest='train_file', default='')
    parser.add_argument('--cv-catalog-file', type=str, dest='cv_file', default='')
    parser.add_argument('--train-ratio', type=float, dest='train_ratio', default=0.8)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    train_ratio = (args.train_ratio if len(args.cv_file) > 0 else 1.0)
    train_catalog, cv_catalog = make_catalog(args.input_dir, train_ratio)

    if len(args.train_file) > 0 and len(train_catalog['dataset']) > 0:
        train_dir = os.path.split(args.train_file)[0]
        if len(train_dir) > 0 and (not os.path.exists(train_dir)):
            os.makedirs(train_dir)
        with open(args.train_file, 'w') as fp:
            json.dump(train_catalog, fp, sort_keys=True, ensure_ascii=False, indent=2)

    if len(args.cv_file) > 0 and len(cv_catalog['dataset']) > 0:
        cv_dir = os.path.split(args.train_file)[0]
        if len(cv_dir) > 0 and (not os.path.exists(cv_dir)):
            os.makedirs(cv_dir)
        with open(args.cv_file, 'w') as fp:
            json.dump(cv_catalog, fp, sort_keys=True, ensure_ascii=False, indent=2)

    print('done')
