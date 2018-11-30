#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: utils.py
# @time: 18-11-27下午3:32
import tensorflow.examples.tutorials.mnist.input_data as input_data
from config.config import MNIST_DATA_PATH
import scipy
import numpy as np


def load_mnist():
    # 下载并加载mnist数据
    mnist = input_data.read_data_sets(MNIST_DATA_PATH, one_hot=True)
    train_label = mnist.train.labels
    test_label = mnist.test.labels
    train_num, _ = train_label.shape
    test_num, _ = test_label.shape
    return mnist, train_num, test_num


def save_images(images, size, path):
    # images = (images + 0.1) / 2  # inverse_transform
    return scipy.misc.imsave(path, merge_images(images, size))


def merge_images(images, size):
    """
    合并图片
    :param images:
    :param size:
    :return:
    """
    h, w = images.shape[1], images.shape[2]
    re_images = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        re_images[j * h:(j + 1) * h, i * w:(i + 1) * w, :] = image
    return re_images
