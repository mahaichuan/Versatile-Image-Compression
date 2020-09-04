from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from math import log
import numpy as np
from PIL import Image
import basic_DL_op

def lstm_logic(x, c):

    i = tf.sigmoid(x)
    f = tf.sigmoid(x)
    o = tf.sigmoid(x)
    g = tf.tanh(x)

    c = f*c+i*g

    h = o*tf.tanh(c)


    return c, h

def lstm_layer(x, h, c, in_num, out_num):

    # the first layer: input

    w = basic_DL_op.weight_variable('conv1', [3, 3, in_num, out_num], 0.01)

    x = basic_DL_op.conv2d(x, w)

    # the first layer: state

    w = basic_DL_op.weight_variable('conv2', [3, 3, out_num, out_num], 0.01)

    h = basic_DL_op.conv2d(h, w)

    b = basic_DL_op.bias_variable('bias', [out_num])



    c, h = lstm_logic(x + h + b, c)


    return c, h


def context_single_band(x1, h1, c1, h2, c2, h3, c3, bit_map):

    with tf.variable_scope('LSTM_' + str(1)):
        c1, h1 = lstm_layer(x1, h1, c1, 1, int(32*bit_map))

    with tf.variable_scope('LSTM_' + str(2)):
        c2, h2 = lstm_layer(h1, h2, c2, int(32*bit_map), int(32 * bit_map))

    with tf.variable_scope('LSTM_' + str(3)):
        c3, h3 = lstm_layer(h2, h3, c3, int(32 * bit_map), 1)

    return h1, c1, h2, c2, h3, c3


def context_single_band_reuse(x1, h1, c1, h2, c2, h3, c3, bit_map):

    with tf.variable_scope('LSTM_' + str(1), reuse=True):
        c1, h1 = lstm_layer(x1, h1, c1, 1, int(32*bit_map))

    with tf.variable_scope('LSTM_' + str(2), reuse=True):
        c2, h2 = lstm_layer(h1, h2, c2, int(32*bit_map), int(32 * bit_map))

    with tf.variable_scope('LSTM_' + str(3), reuse=True):
        c3, h3 = lstm_layer(h2, h3, c3, int(32 * bit_map), 1)

    return h1, c1, h2, c2, h3, c3

def deconv_layer(x):

    x_shape = x.get_shape().as_list()

    kernel = basic_DL_op.weight_variable('deconv', [3, 3, x_shape[3], x_shape[3]], 0.01)

    x = tf.nn.conv2d_transpose(x, kernel, output_shape=[x_shape[0], int(x_shape[1]*2), int(x_shape[2]*2), x_shape[3]], strides=[1, 2, 2, 1], padding="SAME")

    return x



def context_all(LL, HL_collection, LH_collection, HH_collection, bit_map, static_QP):

    c_HL = []
    c_LH = []
    c_HH = []

    x_shape = LL.get_shape().as_list()

    h1 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], int(32 * bit_map)], dtype=tf.float32)
    c1 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], int(32 * bit_map)], dtype=tf.float32)

    h2 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], int(32 * bit_map)], dtype=tf.float32)
    c2 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], int(32 * bit_map)], dtype=tf.float32)

    h3 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], 1], dtype=tf.float32)
    c3 = tf.zeros(shape=[x_shape[0], x_shape[1], x_shape[2], 1], dtype=tf.float32)


    h1, c1, h2, c2, h3, c3 = context_single_band(LL / static_QP, h1, c1, h2, c2, h3, c3, bit_map)

    c_HL.append(h3)

    for j in range(4):

        i = 4 - 1 - j

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(HL_collection[i]/ static_QP, h1, c1, h2, c2, h3, c3, bit_map)

        c_LH.append(h3)

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(LH_collection[i]/ static_QP, h1, c1, h2, c2, h3, c3, bit_map)

        c_HH.append(h3)

        h1, c1, h2, c2, h3, c3 = context_single_band_reuse(HH_collection[i]/ static_QP, h1, c1, h2, c2, h3, c3, bit_map)

        with tf.variable_scope('Deconv_h1' + str(j)):
            h1 = deconv_layer(h1)
        with tf.variable_scope('Deconv_c1' + str(j)):
            c1 = deconv_layer(c1)
        with tf.variable_scope('Deconv_h2' + str(j)):
            h2 = deconv_layer(h2)
        with tf.variable_scope('Deconv_c2' + str(j)):
            c2 = deconv_layer(c2)
        with tf.variable_scope('Deconv_h3' + str(j)):
            h3 = deconv_layer(h3)
        with tf.variable_scope('Deconv_c3' + str(j)):
            c3 = deconv_layer(c3)

        c_HL.append(h3)


    return c_HL, c_LH, c_HH