# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import argparse
import math
import pandas as pd
import cv2
import json
import jsonschema


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
    labels_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    width, height, _ = labels_image.shape
    labels_gray = cv2.cvtColor(labels_image, cv2.COLOR_BGR2GRAY)
    #_, labels_binary = cv2.threshold(labels_gray, 0, 255, cv2.THRESH_BINARY)
    #labels_binary = cv2.bitwise_not(labels_binary)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(labels_gray)

    for stat, centroid in zip(stats, centroids):
        b_x, b_y, b_w, b_h, _ = stat[0], stat[1], stat[2], stat[3], stat[4]
        c_x, c_y = int(math.floor(centroid[1])), int(math.floor(centroid[0]))
        # 面積が極端に小さい・大きい領域は誤検出のため除外
        if b_w * b_h <= (width * 2) or b_w * b_h >= (width * height  * 0.95):
            continue
        # 重心位置の画素からクラス値を決定
        r, g, b = labels_image[c_x][c_y]
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
def make_catalog(input_dir, output_dir):
    dataset = []
    camera_image_dir = os.path.join(input_dir, 'Single')
    label_image_dir = os.path.join(input_dir, 'mask_label', 'single')

    # クラスとRGB値の関係テーブルをロード
    df_items = pd.read_csv(os.path.join(label_image_dir, '..', 'item_table.csv'),
                           encoding='cp932')
    # ラベル画像を起点に情報を収集
    count = 1
    for path in find_label_images(label_image_dir):
        sys.stdout.write('\r{0} parse : {1}'.format(count, path))
        item = make_catalog_item(camera_image_dir, label_image_dir,
                                 path, df_items)
        dataset.append(item)
        count += 1
    return {'dataset': dataset}


def parse_arguments():
    usage = 'make training dataset catalog (sample code)'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--input-dir', type=str, dest='input_dir', required=True)
    parser.add_argument('--output-dir', type=str, dest='output_dir', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    catalog = make_catalog(args.input_dir, args.output_dir)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'train_dataset_catalog_sample.json'), 'w') as fp:
        json.dump(catalog, fp, sort_keys=True, ensure_ascii=False, indent=2)
    print('\nfinished')
