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

######################## lifting 97 forward and inverse transform

lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781, 1.149604398860241] # bior4.4
# lifting_coeff = [-12993./8193., -434./8193., 7233./8193., 3633./8193., 5038./8193., 6659./8193.] # OPENJPEG
# lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 1.230174104914001, 0.812893066115961] # CDF9/7

trainable_set = True

def p_block(x):

    w = basic_DL_op.weight_variable('conv1',[3,3,1,16],0.01)
    b = basic_DL_op.bias_variable('bias1', [16])
    tmp = basic_DL_op.conv2d_pad(x,w) + b

    conv1 = tmp

    tmp = tf.nn.tanh(tmp)
    w = basic_DL_op.weight_variable('conv2', [3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias2', [16])
    tmp = basic_DL_op.conv2d_pad(tmp, w) + b

    tmp = tf.nn.tanh(tmp)
    w = basic_DL_op.weight_variable('conv3', [3, 3, 16, 16], 0.01)
    b = basic_DL_op.bias_variable('bias3', [16])
    tmp = basic_DL_op.conv2d_pad(tmp, w) + b

    tmp = conv1 + tmp

    w = basic_DL_op.weight_variable('conv4', [3, 3, 16, 1], 0.01)
    b = basic_DL_op.bias_variable('bias4', [1])
    out = basic_DL_op.conv2d_pad(tmp, w) + b

    return out

def lifting97_forward(L, H):

    # first lifting step

    # with tf.variable_scope('p_1'):
    #
    #     L_net = p_block(L/256.)*256.

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1])

    w = tf.get_variable('cdf97_0', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv2d(tmp, w, strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):

        L_net = p_block(skip/256.)*256.

    H = H + skip + L_net * 0.1



    # with tf.variable_scope('u_1'):
    #
    #     H_net = p_block(H/256.)*256.

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1])

    w = tf.get_variable('cdf97_1', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv2d(tmp, w, strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):

        H_net = p_block(skip/256.)*256.

    L = L + skip + H_net * 0.1

    # second lifting step

    # with tf.variable_scope('p_2'):
    #
    #     L_net = p_block(L/256.)*256.

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1])

    w = tf.get_variable('cdf97_2', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv2d(tmp, w, strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):

        L_net = p_block(skip/256.)*256.

    H = H + skip + L_net * 0.1



    # with tf.variable_scope('u_2'):
    #
    #     H_net = p_block(H/256.)*256.

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1])

    w = tf.get_variable('cdf97_3', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv2d(tmp, w, strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):

        H_net = p_block(skip/256.)*256.

    L = L + skip + H_net * 0.1

    # scaling step

    n_h = tf.get_variable('n_h', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h * 0.1

    H = H * n_h

    n_l = tf.get_variable('n_l', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l * 0.1

    L = L * n_l

    return L, H


def lifting97_inverse(L, H):

    # scaling step

    n_h = tf.get_variable('n_h', initializer=0.0, trainable=trainable_set)

    n_h = lifting_coeff[4] + n_h * 0.1

    H = H / n_h

    n_l = tf.get_variable('n_l', initializer=0.0, trainable=trainable_set)

    n_l = lifting_coeff[5] + n_l * 0.1

    L = L / n_l

    # second lifting step

    # with tf.variable_scope('u_2'):
    #     H_net = p_block(H / 256.) * 256.

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[3]],
               [lifting_coeff[3]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1])

    w = tf.get_variable('cdf97_3', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv2d(tmp, w, strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_2'):
        H_net = p_block(skip / 256.) * 256.

    L = L - skip - H_net * 0.1



    # with tf.variable_scope('p_2'):
    #     L_net = p_block(L / 256.) * 256.

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[2]],
               [lifting_coeff[2]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1])

    w = tf.get_variable('cdf97_2', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv2d(tmp, w, strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_2'):
        L_net = p_block(skip / 256.) * 256.

    H = H - skip - L_net * 0.1

    # first lifting step

    # with tf.variable_scope('u_1'):
    #     H_net = p_block(H / 256.) * 256.

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])

    tmp = tf.pad(H, paddings=paddings, mode="REFLECT")

    initial = [[lifting_coeff[1]],
               [lifting_coeff[1]],
               [0.0]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1])

    w = tf.get_variable('cdf97_1', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv2d(tmp, w, strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('u_1'):
        H_net = p_block(skip / 256.) * 256.

    L = L - skip - H_net * 0.1


    # with tf.variable_scope('p_1'):
    #     L_net = p_block(L / 256.) * 256.

    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])

    tmp = tf.pad(L, paddings=paddings, mode="REFLECT")

    initial = [[0.0],
               [lifting_coeff[0]],
               [lifting_coeff[0]]]

    initial = tf.reshape(initial, shape=[3, 1, 1, 1])

    w = tf.get_variable('cdf97_0', initializer=initial, trainable=trainable_set)

    skip = tf.nn.conv2d(tmp, w, strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('p_1'):
        L_net = p_block(skip / 256.) * 256.

    H = H - skip - L_net * 0.1

    return L, H
