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
import cv2
import json

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from yolo import *
from bounding_box import *
from image_process import *

# configurations
learning_schedules = {
    '0'    : 1e-5,
    '500'  : 1e-4,
    '10000': 1e-5,
    '20000': 1e-6
}
momentum = 0.9
weight_decay = 0.005

DATASET_PATH = os.path.join('.', 'detector_dataset.npz')
FIRST_MODEL_PATH = os.path.join('.', 'detector_first.model')


def parse_item_of_dataset(item):
    image = Image(item['color_image_path'], INPUT_SIZE, INPUT_SIZE)
    bounding_boxes = [Box(x=float(box['x']), y=float(box['y']),
                            width=float(box['width']), height=float(box['height']),
                            clazz=int(box['class'])) for box in item['bounding_boxes']]
    ground_truth = GroundTruth(width=image.real_width, height=image.real_height,
        bounding_boxes=bounding_boxes)
    return {'image': image, 'ground_truth': ground_truth}

def load_dataset(catalog_file):
    with open(os.path.join(catalog_file), 'r') as fp:
        catalog = json.load(fp)
    dataset = filter(lambda item: item['bounding_boxes'] != [], catalog['dataset'])
    items = [parse_item_of_dataset(item) for item in dataset]
    images = np.asarray([item['image'] for item in items])
    ground_truths = np.asarray([item['ground_truth'] for item in items])
    return images, ground_truths

def prepare_dataset(args):
    print('prepare dataset: train:%s cv:%s' %
        (args.train_catalog_file, args.cv_catalog_file))
    train_images, train_truths = load_dataset(args.train_catalog_file)
    cv_images, cv_truths = load_dataset(args.cv_catalog_file)
    print('save dataset: train images={} ground_truths={}, cv images={} ground_truths={}'
        .format(train_images.shape, train_truths.shape, cv_images.shape, cv_truths.shape))
    np.savez(DATASET_PATH, train_images=train_images, train_ground_truths=train_truths,
        cv_images=cv_images, cv_ground_truths=cv_truths)

def initialize_model(args):
    def copy_conv_layer(src, dst):
        for i in range(1, N_CNN_LAYER+1):
            src_layer = eval('src.conv%d' % i)
            dst_layer = eval('dst.conv%d' % i)
            dst_layer.W = src_layer.W
            dst_layer.b = src_layer.b

    print('initialize model: input:%s output:%s' % \
        (args.input_model_file, args.output_model_file))
    classifier_model = YoloClassifier(args.gpu)
    chainer.serializers.load_npz(args.input_model_file, classifier_model)
    detector_model = YoloDetector(args.gpu)

    '''
    serializers.save_npz()実行時の
    "ValueError: uninitialized parameters cannot be serialized" を回避するために
    ダミーデータでの順伝播を実行する
    '''
    dummy_image = chainer.Variable(np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32))
    detector_model.forward(dummy_image)
    copy_conv_layer(classifier_model, detector_model)
    chainer.serializers.save_npz(args.output_model_file, detector_model)


def parse_ground_truth(box, real_width, real_height):
    tx = (box.left + (box.width / 2.0)) * INPUT_SIZE / real_width
    ty = (box.top  + (box.height / 2.0)) * INPUT_SIZE / real_height
    tw = box.width * INPUT_SIZE / real_width
    th = box.height * INPUT_SIZE / real_height

    grid_size = INPUT_SIZE / N_GRID
    active_grid_cell = {
        'x': int(math.modf(tx / grid_size)[1]),
        'y': int(math.modf(ty / grid_size)[1])
    }
    norm_truth = { # [0..1] に正規化
        'x' : math.modf(tx / grid_size)[0],
        'y' : math.modf(ty / grid_size)[0],
        'w' : tw / INPUT_SIZE,
        'h' : th / INPUT_SIZE
    }
    one_hot_clazz = np.eye(N_CLASSES)[np.array(box.clazz)]

    # detection layerのテンソルに変換
    tensor = np.zeros(((5*N_BOXES)+N_CLASSES, N_GRID, N_GRID)).astype(np.float32)
    tensor[:5, active_grid_cell['y'], active_grid_cell['x']] \
        = [norm_truth['x'], norm_truth['y'], norm_truth['w'], norm_truth['h'], 1.0]
    tensor[5:, active_grid_cell['y'], active_grid_cell['x']] = one_hot_clazz
    return tensor

def make_ground_truth_tensor(ground_truth):
    each_tensors = [parse_ground_truth(box, ground_truth.width, ground_truth.height)
                        for box in ground_truth.bounding_boxes]
    return reduce(lambda x, y: x + y, each_tensors)

def final_detection(grid_box, real_width, real_height):
    grid_size = INPUT_SIZE / N_GRID
    box = Box(x=((grid_box.left * grid_size * real_width / INPUT_SIZE) - (grid_box.width * real_width / 2)),
              y=((grid_box.top * grid_size * real_height / INPUT_SIZE) - (grid_box.height * real_height / 2)),
              width=grid_box.width * real_width,
              height=grid_box.height * real_height,
              clazz=grid_box.clazz,
              objectness=grid_box.objectness)
    box.x = max(0., box.left)
    box.y = max(0., box.top)
    box.width = min(box.width, real_width - box.left)
    box.height = min(box.height, real_height - box.top)
    return box

def evaluate(detections, ground_truths):
    positives = {str(i): {'true': 0, 'false': 0}  for i in range(0, N_CLASSES)}

    for detection, ground_truth in zip(detections, ground_truths):
        for box in detection:
            if Box.correct(box, ground_truth.bounding_boxes):
                positives[str(box.clazz)]['true'] += 1
            else:
                positives[str(box.clazz)]['false'] += 1
    return positives

def average_precisions(positives):
    def precision(tp, fp):
        if tp == 0 and fp == 0:
            return 0
        else:
            return float(tp) / (tp + fp)
    return {str(p[0]): precision(p[1]['true'], p[1]['false']) for p in positives.items()}

def one_epoch_train(model, optimizer, images, ground_truths, batch_size, epoch):
    n_train = len(ground_truths)
    image_tensors  = np.asarray([image.image for image in images]).transpose(0,3,1,2) / 255.
    ground_truth_tensors = np.asarray([make_ground_truth_tensor(truth) for truth in ground_truths])

    loss = 0.
    detections = []
    for count in six.moves.range(0, n_train, batch_size):
        ix = np.arange(count, min(count+batch_size, n_train))
        xs = chainer.Variable(xp.asarray(image_tensors[ix]).astype(np.float32))
        ts = chainer.Variable(xp.asarray(ground_truth_tensors[ix]).astype(np.float32))

        model.train = True
        iteration = (epoch - 1) * batch_size + count
        if str(iteration) in learning_schedules:
            optimizer.lr = learning_schedules[str(iteration)]
        optimizer.update(model, xs, ts)
        loss += model.loss.data * len(ix) / n_train
        scaled_detections = [[final_detection(box, truth.width, truth.height)
                              for box in detection]
                             for detection, truth in zip(model.detections, ground_truths)]
        detections += scaled_detections

    count = 1
    for detection, truth in zip(detections, ground_truths):
        for box in detection:
            print(count, box, truth.bounding_boxes[0])
        count += 1

    positives = evaluate(detections, ground_truths)
    aps = average_precisions(positives)
    map = sum(aps.values()) / len(aps.values())
    return loss, map

def one_epoch_cv(model, optimizer, images, ground_truths):
    n_valid = len(ground_truths)
    image_tensors  = np.asarray([image.image for image in images]).transpose(0,3,1,2) / 255.
    ground_truth_tensors = np.asarray([make_ground_truth_tensor(truth) for truth in ground_truths])

    loss = 0.
    detections = []
    for count in six.moves.range(0, n_valid, 10):
        ix = np.arange(count, min(count+10, n_valid))
        xs = chainer.Variable(xp.asarray(image_tensors[ix]).astype(np.float32))
        ts = chainer.Variable(xp.asarray(ground_truth_tensors[ix]).astype(np.float32))

        model.train = False
        model(xs, ts)
        loss += model.loss.data * len(ix) / n_valid
        scaled_detections = [[final_detection(box, truth.width, truth.height)
                              for box in detection]
                             for detection, truth in zip(model.detections, ground_truths)]
        detections += scaled_detections

    count = 1
    for detection, truth in zip(detections, ground_truths):
        for box in detection:
            print(count, box, truth.bounding_boxes[0])
        count += 1

    positives = evaluate(detections, ground_truths)
    aps = average_precisions(positives)
    map = sum(aps.values()) / len(aps.values())
    return loss, map

def train_model(args):
    print('train model: gpu:%d epoch:%d batch_size:%d init_model:%s init_state:%s' % \
        (args.gpu, args.n_epoch, args.batch_size, args.init_model_file, args.init_state_file))

    model = YoloDetector(args.gpu)
    if len(args.init_model_file) > 0:
        chainer.serializers.load_npz(args.init_model_file, model)

    optimizer = chainer.optimizers.MomentumSGD(
        lr=learning_schedules['0'], momentum=momentum)
    optimizer.setup(model)
    if len(args.init_state_file) > 0:
        chainer.serializers.load_npz(args.init_state_file, optimizer)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    optimizer.use_cleargrads()

    logs = []
    dataset = np.load(DATASET_PATH)
    train_images = dataset['train_images']
    train_truths = dataset['train_ground_truths']
    cv_images = dataset['cv_images']
    cv_truths = dataset['cv_ground_truths']
    print('number of dataset: train:%d cv:%d' % (len(train_truths), len(cv_truths)))

    for epoch in six.moves.range(1, args.n_epoch+1):
        train_loss, train_map = one_epoch_train(
            model, optimizer, train_images, train_truths, args.batch_size, epoch)
        cv_loss, cv_map = one_epoch_cv(
            model, optimizer, cv_images, cv_truths)
        print('epoch:%d trian loss:%f train map:%f cv loss:%f cv map:%f' %
            (epoch, train_loss, train_map, cv_loss, cv_map))
        logs.append({
            'epoch': str(epoch),
            'train_loss': str(train_loss), 'train_map': str(train_map),
            'cv_loss': str(cv_loss), 'cv_map': str(cv_map)
        })
        if (epoch % 10) == 0:
            chainer.serializers.save_npz('detector_epoch{}.model'.format(epoch), model)
            chainer.serializers.save_npz('detector_epoch{}.state'.format(epoch), optimizer)

    chainer.serializers.save_npz('detector_final.model', model)
    chainer.serializers.save_npz('detector_final.state', optimizer)
    with open('detector_train_log.json', 'w') as fp:
        json.dump({'epoch': str(args.n_epoch), 'batch_size': str(args.batch_size), 'logs': logs},
            fp, sort_keys=True, ensure_ascii=False, indent=2)


def parse_arguments():
    usage = 'make training dataset catalog (sample code)'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--action', '-a', type=str, dest='action', required=True)
    # prepare options
    parser.add_argument('--train-catalog-file', type=str, dest='train_catalog_file', default='')
    parser.add_argument('--cv-catalog-file', type=str, dest='cv_catalog_file', default='')
    # initialize options
    parser.add_argument('--input-model-file', type=str, dest='input_model_file', default='')
    parser.add_argument('--output-model-file', type=str, dest='output_model_file', default=FIRST_MODEL_PATH)
    # train/predict options
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--batchsize', '-b', type=int, dest='batch_size', default=20)
    parser.add_argument('--epoch', '-e', type=int, dest='n_epoch', default=1)
    parser.add_argument('--init-model-file', type=str, dest='init_model_file', default=FIRST_MODEL_PATH)
    parser.add_argument('--init-state-file', type=str, dest='init_state_file', default='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy

    if args.action == 'prepare':
        prepare_dataset(args)
    elif args.action == 'initialize':
        initialize_model(args)
    elif args.action == 'train':
        train_model(args)
    print('done')
