# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import argparse
import math
import numpy as np
import pandas as pd
import csv
import cv2

'''
ビットマップファイルを探索する
'''
def find_label_bitmap_files(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f == str('label.bmp'):
                yield os.path.join(root, f)

'''
ラベルの矩形領域とクラス値を抽出する
'''
def extract_label_box(path, df_items):
    # 抽出結果を格納するデータフレームを作成
    columns = ['x', 'y', 'w', 'h', 'class']
    df_boxes = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'class'])

    # ラベルの矩形領域と重心位置を取得
    # TODO: このロジックで複数ラベルに対応できるかは要確認
    labels_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    width, height, _ = labels_image.shape
    labels_gray = cv2.cvtColor(labels_image, cv2.COLOR_BGR2GRAY)
    _, labels_binary = cv2.threshold(labels_gray, 0, 255, cv2.THRESH_BINARY)
    #labels_binary = cv2.bitwise_not(labels_binary)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(labels_gray)

    for stat, centroid in zip(stats, centroids):
        r_x, r_y, r_w, r_h, _ = stat[0], stat[1], stat[2], stat[3], stat[4]
        c_x, c_y = int(math.floor(centroid[1])), int(math.floor(centroid[0]))
        # 面積が極端に小さい・大きい領域は誤検出のため除外
        if r_w * r_h <= (width * 2) or r_w * r_h >= (width * height  * 0.95):
            continue
        # 重心位置の画素からクラス値を決定
        r, g, b = labels_image[c_x][c_y]
        clazz = int(df_items[(df_items['r'] == r) & (df_items['g'] == g) & (df_items['b'] == b)]['class'].values[0])
        df_boxes = df_boxes.append(pd.Series([r_x, r_y, r_w, r_h, clazz], index=columns),
                                   ignore_index=True)
    return df_boxes

'''
ラベル画像の矩形領域とクラス値を抽出しCSVファイルに書き出す
'''
def extract(root_dir):
    path = os.path.join(root_dir, 'item_table.csv')
    df_items = pd.read_csv(path, encoding='cp932')

    count = 1
    for path in find_label_bitmap_files(root_dir):
        sys.stdout.write('\r{0} extract : {1} ...'.format(count, path))
        df_boxes = extract_label_box(path, df_items)
        dir_path, _ = os.path.split(path)
        path = os.path.join(dir_path, 'label_box.csv')
        df_boxes.to_csv(path, encoding='cp932', #line_terminator='\r\n',
                        header=True, index=False, mode='w')
        count += 1
    print('\nfinished')

def parse_arguments():
    usage = 'extract label boxes and class values'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--label-dir', type=str, dest='label_dir', required=True)
    args = parser.parse_args()
    return args.label_dir

if __name__ == '__main__':
    label_dir = parse_arguments()
    extract(label_dir)
