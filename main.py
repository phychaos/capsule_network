#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: main.py
# @time: 18-11-26下午2:06
# from model.CapsNet import CapsuleNet
import copy
import os

import tensorflow as tf
from config.config import MODEL_PATH, OUTPUT_PATH
from config.parameter import HyperParameter as para
from core.utils import load_mnist, save_images

from model.CapsNet import CapsuleNet
import numpy as np


def main(_):
    """
    训练模型
    :return:
    """
    caps_model = CapsuleNet(is_training=True)
    sv = tf.train.Supervisor(logdir=MODEL_PATH, graph=caps_model.graph, global_step=caps_model.global_step,
                             summary_op=caps_model.summary)
    mnist_data, train_num, test_num = load_mnist()
    x_train = mnist_data.train.images
    y_train = mnist_data.train.labels
    x_test = mnist_data.test.images
    y_test = mnist_data.test.labels
    with sv.managed_session() as sess:
        max_eval = 0
        for epoch in range(para.num_epoch):
            # 训练
            train_loss, train_acc = train_eval(sess, caps_model, x_train, y_train, train_num, is_training=True)
            print("train\tepoch\t{},\tloss\t{},\tacc\t{}".format(epoch, train_loss, train_acc))
            # 测试
            test_loss, test_acc = train_eval(sess, caps_model, x_test, y_test, test_num, is_training=False)
            print("eval\tepoch\t{},\tloss\t{},\tacc\t{}".format(epoch, test_loss, test_acc))

            if test_acc > max_eval:
                max_eval = np.mean(test_acc)
                global_step = sess.run(caps_model.global_step)
                sv.saver.save(sess, MODEL_PATH + '/model_epoch_{}_gs_{}'.format(epoch, global_step))


def train_eval(sess, caps_model, x_data, y_data, num, is_training=False):
    """
    训练评估模型
    :param sess:
    :param caps_model:
    :param x_data:
    :param y_data:
    :param num:
    :param is_training:
    :return:
    """
    total_loss = []
    total_acc = []
    for kk in range(int(num / para.batch_size)):
        start = kk * para.batch_size
        end = (kk + 1) * para.batch_size
        x = x_data[start:end, :]
        y = y_data[start:end, :]
        feed_dict = {caps_model.x: x, caps_model.y: y}
        if is_training:
            _, loss, acc = sess.run([caps_model.train_op, caps_model.loss, caps_model.accuracy], feed_dict=feed_dict)
        else:
            loss, acc = sess.run([caps_model.loss, caps_model.accuracy], feed_dict=feed_dict)
        total_acc.append(acc)
        total_loss.append(loss)
    loss_value = round(float(np.mean(total_loss)), 4)
    acc_value = round(float(np.mean(total_acc)), 4)
    return loss_value, acc_value


def reconstruct_image():
    mnist_data, train_num, test_num = load_mnist()
    x_test = mnist_data.test.images[:para.batch_size]
    size = 5
    caps_model = CapsuleNet(is_training=False)
    with caps_model.graph.as_default():
        sv = tf.train.Supervisor(logdir=MODEL_PATH)
        with sv.managed_session() as sess:
            checkpoint_path = tf.train.latest_checkpoint(MODEL_PATH)
            sv.saver.restore(sess, checkpoint_path)

            recon_image = sess.run(caps_model.decoded, feed_dict={caps_model.x: x_test})
            recon_image = np.reshape(recon_image, (para.batch_size, 28, 28, 1))
            x_test = np.reshape(x_test, (para.batch_size, 28, 28, 1))
            for ii in range(5):
                start = ii * size * size
                end = (ii + 1) * size * size
                recon_filename = os.path.join(OUTPUT_PATH, 'recon_image_{}.png'.format(ii + 1))
                save_images(recon_image[start:end, :], [size, size], recon_filename)

                test_filename = os.path.join(OUTPUT_PATH, 'test_image_{}.png'.format(ii + 1))
                save_images(x_test[start:end, :], [size, size], test_filename)


if __name__ == "__main__":
    # tf.app.run()
    reconstruct_image()
