from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from math import log
import numpy as np
from PIL import Image

#################  we define the basic DL operations here

def weight_variable(name,shape,std):

  initial = tf.truncated_normal(shape, stddev=std)

  return tf.get_variable(name, initializer=initial)


def bias_variable(name, shape):

  initial = tf.constant(0., shape=shape)

  return tf.get_variable(name, initializer=initial)


def conv2d(x, W, stride=1):

  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def conv2d_pad(x, W, stride=1):

    paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])

    x = tf.pad(x, paddings=paddings,mode="REFLECT")

    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
#
# def conv3d(x,W):
#
#     return tf.nn.conv3d(input=x, filter=W, strides=[1,1,1,1,1], padding='SAME')