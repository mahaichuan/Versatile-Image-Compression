import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import shutil

# basic op of deep learning
def weight_variable(name,shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.get_variable(name, initializer=initial)

def bias_variable(name,shape):
  initial = tf.constant(0., shape=shape)
  return tf.get_variable(name, initializer=initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# basic residual block of EDEH
def resBlock(x, channels=16):

    w = weight_variable('conv1',[3,3,channels,channels])
    b = bias_variable('bias1',[channels])

    tmp = conv2d(x,w) + b

    tmp = tf.nn.relu(tmp)


    w = weight_variable('conv2',[3, 3, channels, channels])
    b = bias_variable('bias2',[channels])

    tmp = conv2d(tmp,w) + b

    return x + tmp

# Network of EDEH

def EDEH(x, num_layers=6,feature_size=64,output_channels=1):

    in_x = x

    # ////////////////   conv: before residual blocks

    w = weight_variable('conv1',[3,3,1,feature_size])
    b = bias_variable('bias1',[feature_size])

    x = conv2d(x,w) + b
    conv_1 = x


    # //////////////    residual blocks

    for i in range(num_layers):
        with tf.variable_scope('resBlock_' + str(i)):
            x = resBlock(x, feature_size)

    # /////////////    One more convolution, and then we add the output of our first conv layer
    w = weight_variable('conv2',[3, 3, feature_size, feature_size])
    b = bias_variable('bias2',[feature_size])
    x = conv2d(x,w) + b
    x = x + conv_1

    # //////////// one more convolution after direct addition
    w = weight_variable('conv3',[3,3,feature_size,output_channels])
    b = bias_variable('bias3',[output_channels])
    x = conv2d(x,w) + b

    x = x + in_x

    return x