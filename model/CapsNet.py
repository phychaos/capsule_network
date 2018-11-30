#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: CapsNet.py
# @time: 18-11-26下午2:07
import tensorflow as tf
from config.parameter import HyperParameter as para
from tensorflow.contrib.layers import fully_connected
import numpy as np


class CapsuleNet(object):
    def __init__(self, is_training=False):
        """
        胶囊神经网络初始化
        :param is_training: 是否训练
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input_x')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_y')
            self.build_model(is_training)

    def build_model(self, is_training):
        """
        构建模型架构
        :param is_training:
        :return:
        """

        inputs = tf.reshape(self.x, [-1, 28, 28, 1])  # batch * weight * height * channels

        convolution = self.convolution_layer(inputs)  # 卷积层 batch * 20 * 20 * 256
        primary_caps = self.primary_caps(convolution)  # primary_caps batch * 32*6*6 * 8
        digit_caps = self.digit_caps(primary_caps, para.num_iter, para.num_caps)  # digit_caps batch * 10 * 16

        logits = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=2) + para.eps)
        probs = tf.nn.softmax(logits)
        preds = tf.argmax(probs, axis=1)
        correct_prediction = tf.equal(preds, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        targets = tf.argmax(self.y, axis=1) if is_training else preds
        self.decoded = self.reconstruction(digit_caps, targets)

        if not is_training:
            return
        m_loss = self.margin_loss(probs, self.y)
        r_loss = self.reconstruction_loss(self.x, self.decoded)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.loss = m_loss + 0.0005 * r_loss
        self.train_op = tf.train.AdamOptimizer(para.lr).minimize(self.loss, global_step=self.global_step)
        # Summary
        tf.summary.scalar('margin_loss', m_loss)
        tf.summary.scalar('reconstruction_loss', r_loss)
        tf.summary.scalar('total_loss', self.loss)
        self.summary = tf.summary.merge_all()
        return

    @staticmethod
    def convolution_layer(inputs):
        """
        卷积层 核size=(9 * 9) filters=256  ==> batch * 20 * 20 * 256
        :param inputs: batch * weight * height * channels
        :return:
        """
        with tf.variable_scope("convolution_layer"):
            weight = tf.Variable(tf.truncated_normal([9, 9, 1, 256], stddev=0.1, name='w'))
            bias = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
            convolution = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='VALID')
            convolution = tf.nn.bias_add(convolution, bias, data_format="NHWC")
            convolution = tf.nn.relu(convolution)
        return convolution

    def primary_caps(self, inputs):
        """
        基本胶囊层 batch * 20 * 20 * 256  ==> batch * (32*6*6)  * 8
        :param inputs: batch * 20 * 20 * 256
        :return:
        """
        with tf.variable_scope("primary_caps"):
            weight = tf.Variable(tf.truncated_normal([9, 9, 256, 32 * 8], stddev=0.1, name='w'))
            bias = tf.get_variable('bias', [32 * 8], initializer=tf.constant_initializer(0.0))
            convolution = tf.nn.conv2d(input=inputs, filter=weight, strides=[1, 2, 2, 1], padding='VALID')
            convolution = tf.nn.bias_add(convolution, bias, data_format='NHWC')
            convolution = tf.reshape(convolution, [-1, 32 * 6 * 6, 8])
            primary_caps = self.squash(convolution)
        return primary_caps

    def digit_caps(self, inputs, num_iter, num_caps):
        """
        数字胶囊层
        :param inputs: batch * 1152  * 8
        :param num_iter: 3
        :param num_caps: 10
        :return:
        """
        with tf.variable_scope("digit_caps"):
            u = tf.reshape(inputs, [para.batch_size, 32 * 6 * 6, 1, 1, 8])
            u = tf.tile(u, [1, 1, num_caps, 1, 1])
            b_ij = tf.zeros((32 * 6 * 6, num_caps), name='b')
            w = tf.get_variable(name='weight', shape=[1, 32 * 6 * 6, num_caps, 8, 16],
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            w = tf.tile(w, [para.batch_size, 1, 1, 1, 1])

            u_hat = tf.matmul(u, w)  # u^T *w [1 × 8 ] dot [ 8 × 16] = 1 × 16
            u_hat = tf.reshape(u_hat, [para.batch_size, 32 * 6 * 6, num_caps, 16])
            bias = tf.get_variable('bias', shape=[1, num_caps, 16], initializer=tf.constant_initializer(0.0))
            for r in range(num_iter):
                with tf.variable_scope('routing_iter_' + str(r)):
                    c_ij = tf.nn.softmax(b_ij, dim=-1)
                    c_ij = tf.tile(tf.reshape(c_ij, [1, 32 * 6 * 6, num_caps, 1]), [para.batch_size, 1, 1, 1])

                    s = tf.reduce_sum(tf.multiply(u_hat, c_ij), axis=1) + bias
                    v = self.squash(s)

                    vr = tf.reshape(v, [para.batch_size, 1, num_caps, 16])
                    a = tf.reduce_sum(tf.reduce_sum(tf.multiply(u_hat, vr), axis=0), axis=2)
                    b_ij = b_ij + a
        return v

    @staticmethod
    def squash(inputs):
        """
        激活函数 v = s/|s| * s^2/(1+s^2) < 1
        :param inputs:
        :return:
        """
        square = tf.reduce_sum(tf.square(inputs), axis=-1, keep_dims=True) + para.eps
        scalar_factor = square / (1 + square)
        v = scalar_factor * tf.divide(inputs, tf.sqrt(square))
        return v

    @staticmethod
    def margin_loss(logits, labels, margin=0.9, down_weight=0.5):
        """
        折叶损失函数
        :param logits: batch * 10
        :param labels:
        :param margin:
        :param down_weight:
        :return:
        """
        _margin = 1 - margin
        positive_cost = labels * tf.pow(tf.nn.relu(margin - logits), 2)
        negative_cost = (1 - labels) * tf.pow(tf.nn.relu(logits - _margin), 2)

        margin_loss = positive_cost + down_weight * negative_cost
        margin_loss = tf.reduce_mean(tf.reduce_sum(margin_loss, axis=1))
        return margin_loss

    @staticmethod
    def reconstruction(inputs, targets):
        """
        重构图像
        :param inputs: batch *
        :param targets:
        :return:
        """
        with tf.variable_scope('masking'):
            capsule_mask = tf.one_hot(targets, depth=10, on_value=1.0, off_value=0.0, axis=-1)
            capsule_mask = tf.expand_dims(capsule_mask, axis=-1)
            mask_inputs = tf.reshape(inputs * capsule_mask, [para.batch_size, -1])
        # 重构图像
        with tf.variable_scope('reconstruction'):
            fc_relu1 = fully_connected(inputs=mask_inputs, num_outputs=512, activation_fn=tf.nn.relu)
            fc_relu2 = fully_connected(inputs=fc_relu1, num_outputs=1024, activation_fn=tf.nn.relu)
            fc_sigmoid = fully_connected(inputs=fc_relu2, num_outputs=784, activation_fn=tf.nn.sigmoid)
            assert fc_sigmoid.get_shape() == (para.batch_size, 784)
            recons_images = tf.reshape(fc_sigmoid, shape=(para.batch_size, 28, 28))
        return recons_images

    @staticmethod
    def reconstruction_loss(origin, decoded):
        """
        重建损失函数
        :param origin:
        :param decoded:
        :return:
        """
        origin = tf.reshape(origin, shape=(para.batch_size, -1))
        decoded = tf.reshape(decoded, shape=(para.batch_size, -1))
        r_loss = tf.reduce_mean(tf.square(decoded - origin))
        return r_loss
