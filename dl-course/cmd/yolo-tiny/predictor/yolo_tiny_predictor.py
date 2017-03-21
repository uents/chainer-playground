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
import scipy.io
import cv2
import json

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Function, Link

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from yolo import *
from image_process import *

# configurations
ORIG_WIDTH = 1280
ORIG_HEIGHT = 960

TRAIN_DATASET_PATH = os.path.join('.', 'train_dataset.npz')
INIT_MODEL_PATH = os.path.join('.', 'predictor_init.model')


def parse_ground_truth(truth, orig_width, orig_height):
    def load_ground_truth(truth):
        # YOLOの入力画像の座標系 (始点はオブジェクト中心) に変換
        w = float(truth['width']) * INPUT_SIZE / orig_width
        h = float(truth['height']) * INPUT_SIZE / orig_height
        x = (float(truth['x']) * INPUT_SIZE / orig_width) + (w / 2)
        y = (float(truth['y']) * INPUT_SIZE / orig_height) + (h / 2)
        conf = int(truth['class'])
        return x, y, w, h, conf

    tx, ty, tw, th, tconf = load_ground_truth(truth)
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
    one_hot_confidence_vector = np.eye(N_CLASSES)[np.array(tconf)]

    # detection layerのテンソルに変換
    tensor = np.zeros(((5*N_BOXES)+N_CLASSES, N_GRID, N_GRID)).astype(np.float32)
    tensor[:5, active_grid_cell['y'], active_grid_cell['x']] \
        = [norm_truth['x'], norm_truth['y'], norm_truth['w'], norm_truth['h'], 1.0]
    tensor[5:, active_grid_cell['y'], active_grid_cell['x']] \
        = one_hot_confidence_vector
    return tensor

def make_ground_truth_tensor(truths):
    each_tensors = [parse_ground_truth(t, ORIG_WIDTH, ORIG_HEIGHT) for t in truths]
    return reduce(lambda x,y: x + y, each_tensors)

def prepare_dataset(args):
    print('prepare dataset: catalog:%s' % (args.catalog_file))
    with open(os.path.join(args.catalog_file), 'r') as fp:
        catalog = json.load(fp)
    train_dataset = filter(lambda item: item['bounding_boxes'] != [], catalog['dataset'])
    images = np.asarray([load_image(item['color_image_path'], INPUT_SIZE, INPUT_SIZE) for item in train_dataset])
    ground_truths = np.asarray([make_ground_truth_tensor(item['bounding_boxes']) for item in train_dataset]).astype(np.float32)
    print('save train dataset: images={}, ground_truths={}'.format(images.shape, ground_truths.shape))
    np.savez(TRAIN_DATASET_PATH, images=images, ground_truths=ground_truths)

def initialize_model(args):
    def copy_conv_layer(src, dst):
        for i in range(1, N_CNN_LAYER+1):
            src_layer = eval('src.conv%d' % i)
            dst_layer = eval('dst.conv%d' % i)
            dst_layer.W = src_layer.W
            dst_layer.b = src_layer.b

    print('initialize model: input:%s output:%s' % \
        (args.input_model_file, args.output_model_file))
    cnn_model = YoloTinyCNN()
    chainer.serializers.load_npz(args.input_model_file, cnn_model)
    if args.gpu >= 0: cnn_model.to_gpu()

    predictor_model = YoloTiny()
    if args.gpu >= 0: predictor_model.to_gpu()

    # serializers.save_npz()実行時の
    # "ValueError: uninitialized parameters cannot be serialized" を回避するために
    # ダミーデータでの順伝播を実行する
    dummy_image = Variable(np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32))
    predictor_model.forward(dummy_image)

    copy_conv_layer(cnn_model, predictor_model)
    chainer.serializers.save_npz(args.output_model_file, predictor_model)

def one_epoch_train(model, optimizer, images, ground_truths, batch_size):
    n_train = len(ground_truths)
    perm = np.random.permutation(n_train)

    sum_loss, sum_acc = (0., 0.)
    for count in six.moves.range(0, n_train, batch_size):
        ix = perm[count:count+batch_size]
        xs = chainer.Variable(xp.asarray(images[ix]).astype(np.float32).transpose(0,3,1,2))
        ts = chainer.Variable(xp.asarray(ground_truths[ix]).astype(np.float32))

        model.train = True
        optimizer.update(model, xs, ts)
#        print('mini-batch:%d loss:%f' % ((count/batch_size)+1, model.loss.data))
#        print('mini-batch:%d loss:%f acc:%f' % ((count/batch_size)+1, model.loss.data, model.accuracy.data))
        sum_loss += model.loss.data * len(ix) / n_train
        #sum_acc += model.accuracy.data * len(ix) / n_train
    return sum_loss, sum_acc

def one_epoch_cv(model, optimizer, images, ground_truths):
    n_valid = len(ground_truths)

    sum_loss, sum_acc = (0., 0.)
    for count in six.moves.range(0, n_valid, 10):
        ix = np.arange(count, count+10)
        xs = chainer.Variable(xp.asarray(images[ix]).astype(np.float32).transpose(0,3,1,2))
        ts = chainer.Variable(xp.asarray(ground_truths[ix]).astype(np.int32))

        model.train = False
        model(xs, ts)
        sum_loss += model.loss.data * len(ix) / n_valid
#        sum_acc += model.accuracy.data * len(ix) / n_valid
    return sum_loss, sum_acc

def train_model(args):
    print('train model: gpu:%d epoch:%d batch_size:%d init_model:%s init_state:%s' % \
        (args.gpu, args.n_epoch, args.batch_size, args.init_model_file, args.init_state_file))

    model = YoloTiny()
    if len(args.init_model_file) > 0:
        chainer.serializers.load_npz(args.init_model_file, model)
    if args.gpu >= 0: model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
    if len(args.init_state_file) > 0:
        chainer.serializers.load_npz(args.init_state_file, optimizer)

    logs = []
    train_dataset = np.load(TRAIN_DATASET_PATH)
    images = train_dataset['images']
    ground_truths = train_dataset['ground_truths']

    train_ixs = sorted(random.sample(range(ground_truths.shape[0]), int(ground_truths.shape[0] * 0.8)))
    cv_ixs = sorted(list(set(range(ground_truths.shape[0])) - set(train_ixs)))
    print('number of dataset: train:%d cv:%d' % (len(train_ixs), len(cv_ixs)))

    for epoch in six.moves.range(1, args.n_epoch+1):
        train_loss, train_acc = one_epoch_train(model, optimizer,
            images[train_ixs], ground_truths[train_ixs], args.batch_size)
        cv_loss, cv_acc = one_epoch_cv(model, optimizer,
            images[cv_ixs], ground_truths[cv_ixs])
        print('epoch:%d trian loss:%f train acc:%f cv loss:%f cv acc:%f' %
            (epoch, train_loss, train_acc, cv_loss, cv_acc))
        logs.append({'epoch': str(epoch),
            'train_loss': str(train_loss), 'train_acc': str(train_acc),
            'cv_loss': str(cv_loss), 'cv_acc': str(cv_acc)})
        if (epoch % 10) == 0:
            chainer.serializers.save_npz('predictor_epoch{}.model'.format(epoch), model)
            chainer.serializers.save_npz('predictor_epoch{}.state'.format(epoch), optimizer)

    chainer.serializers.save_npz('predictor_final.model', model)
    chainer.serializers.save_npz('predictor_final.state', optimizer)
    with open('predictor_train_log.json', 'w') as fp:
        json.dump({'epoch': str(args.n_epoch), 'batch_size': str(args.batch_size), 'logs': logs},
            fp, sort_keys=True, ensure_ascii=False, indent=2)


def parse_arguments():
    usage = 'make training dataset catalog (sample code)'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--action', '-a', type=str, dest='action', required=True)
    # prepare options
    parser.add_argument('--catalog-file', type=str, dest='catalog_file')
    # init options
    parser.add_argument('--input-model-file', type=str, dest='input_model_file')
    parser.add_argument('--output-model-file', type=str, dest='output_model_file', default=INIT_MODEL_PATH)
    # train/predict options
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--batchsize', '-b', type=int, dest='batch_size', default=20)
    parser.add_argument('--epoch', '-e', type=int, dest='n_epoch', default=1)
    parser.add_argument('--init-model-file', type=str, dest='init_model_file', default=INIT_MODEL_PATH)
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
