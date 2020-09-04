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

def mask_2D_resiBlock(x, filter_nums):

    w = basic_DL_op.weight_variable('conv1', [3, 3, filter_nums, filter_nums], 0.01)


    mask = [[1,1,1],
            [1,1,0],
            [0,0,0]]

    mask = tf.reshape(mask, shape=[3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, filter_nums, filter_nums])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias1', [filter_nums])

    c = basic_DL_op.conv2d(x, w) + b

    c = tf.nn.relu(c)



    w = basic_DL_op.weight_variable('conv2', [3, 3, filter_nums, filter_nums], 0.01)

    w = w * mask

    b = basic_DL_op.bias_variable('bias2', [filter_nums])

    c = basic_DL_op.conv2d(c, w) + b

    return x + c

def cal_cdf(logits, h, b, a):

    shape = logits.get_shape().as_list()
    logits = tf.reshape(logits, [shape[0], shape[1], shape[2], shape[3], 1])

    logits = tf.matmul(tf.reshape(h[:,:,:,0:3],   [shape[0], shape[1], shape[2], 3,1]), logits)
    logits = logits +  tf.reshape(b[:,:,:,0:3],   [shape[0], shape[1], shape[2], 3,1])
    logits = logits +  tf.reshape(a[:,:,:,0:3],   [shape[0], shape[1], shape[2], 3,1]) * tf.tanh(logits)

    logits = tf.matmul(tf.reshape(h[:,:,:,3:12],  [shape[0], shape[1], shape[2], 3,3]), logits)
    logits = logits +  tf.reshape(b[:,:,:,3:6],   [shape[0], shape[1], shape[2], 3,1])
    logits = logits +  tf.reshape(a[:,:,:,3:6],   [shape[0], shape[1], shape[2], 3,1]) * tf.tanh(logits)

    logits = tf.matmul(tf.reshape(h[:,:,:,12:21], [shape[0], shape[1], shape[2], 3,3]), logits)
    logits = logits +  tf.reshape(b[:,:,:,6:9],   [shape[0], shape[1], shape[2], 3,1])
    logits = logits +  tf.reshape(a[:,:,:,6:9],   [shape[0], shape[1], shape[2], 3,1]) * tf.tanh(logits)

    logits = tf.matmul(tf.reshape(h[:,:,:,21:30], [shape[0], shape[1], shape[2], 3,3]), logits)
    logits = logits +  tf.reshape(b[:,:,:,9:12],  [shape[0], shape[1], shape[2], 3,1])
    logits = logits +  tf.reshape(a[:,:,:,9:12],  [shape[0], shape[1], shape[2], 3,1]) * tf.tanh(logits)

    logits = tf.matmul(tf.reshape(h[:,:,:,30:33], [shape[0], shape[1], shape[2], 1,3]), logits)
    logits = logits +  tf.reshape(b[:,:,:,12:13], [shape[0], shape[1], shape[2], 1,1])

    logits = tf.sigmoid(logits)
    logits = tf.reshape(logits, [shape[0], shape[1], shape[2], shape[3]])

    return logits

def mask_2D_layer(x, static_QP, features = 128, resi_num = 2, para_num = 58):

    x = x / static_QP
    label = x

    # x = tf.stop_gradient(x)

    ################## layer 1, linear

    w = basic_DL_op.weight_variable('conv1', [3, 3, 1, features], 0.01)

    mask = [[1, 1, 1],
            [1, 0, 0],
            [0, 0, 0]]

    mask = tf.reshape(mask, shape=[3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, 1, features])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias1', [features])

    x = basic_DL_op.conv2d(x, w) + b


    conv1 = x

    ################## layers: resi_num resi_block

    for i in range(resi_num):
        with tf.variable_scope('resi_block' + str(i)):

            x = mask_2D_resiBlock(x, features)

    x = conv1 + x

    ################# conv: after skip connection, relu

    w = basic_DL_op.weight_variable('conv2', [3, 3, features, features], 0.01)

    mask = [[1, 1, 1],
            [1, 1, 0],
            [0, 0, 0]]

    mask = tf.reshape(mask, shape=[3, 3, 1, 1])

    mask = tf.tile(mask, multiples=[1, 1, features, features])

    mask = tf.cast(mask, dtype=tf.float32)

    w = w * mask

    b = basic_DL_op.bias_variable('bias2', [features])

    x = basic_DL_op.conv2d(x, w) + b

    x = tf.nn.relu(x)

    ################# convs: 1x1, relu/linear

    w = basic_DL_op.weight_variable('conv3', [1, 1, features, features], 0.01)

    b = basic_DL_op.bias_variable('bias3', [features])

    x = basic_DL_op.conv2d(x, w) + b

    x = tf.nn.relu(x)


    w = basic_DL_op.weight_variable('conv4', [1, 1, features, features], 0.01)

    b = basic_DL_op.bias_variable('bias4', [features])

    x = basic_DL_op.conv2d(x, w) + b

    x = tf.nn.relu(x)


    w = basic_DL_op.weight_variable('conv5', [1, 1, features, para_num], 0.01)

    b = basic_DL_op.bias_variable('bias5', [para_num])

    x = basic_DL_op.conv2d(x, w) + b

    ################# cal the cdf with the output params

    h = tf.nn.softplus( x[:,:,:,0:33] )
    b = x[:,:,:,33:46]
    a = tf.tanh( x[:,:,:,46:58] )

    lower = label - 0.5 / static_QP
    high = label  + 0.5 / static_QP

    lower = cal_cdf(lower, h, b, a)
    high = cal_cdf(high, h, b, a)

    prob = tf.maximum( (high - lower) , 1e-9)

    cross_entropy = -tf.reduce_mean(tf.log(prob))

    return cross_entropy
