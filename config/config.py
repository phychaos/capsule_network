#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: config.py
# @time: 18-11-26下午2:04
import os

PATH = os.getcwd()

DATA_PATH = os.path.join(PATH, 'data')

OUTPUT_PATH = os.path.join(PATH, 'output')

MODEL_PATH = os.path.join(PATH, 'model/ckpt')
# MNIST 数据源
MNIST_DATA_PATH = os.path.join(PATH, 'data/mnist')
