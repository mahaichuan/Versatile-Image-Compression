from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from math import log
import numpy as np
from PIL import Image
import lifting97


############### one-level decomposition of lifting 97

def decomposition(x):

    # step 1: for h
    L = x[:, 0::2, :, :]
    H = x[:, 1::2, :, :]

    L, H = lifting97.lifting97_forward(L, H)



    # step 2: for w, L

    L = tf.transpose(L, [0, 2, 1, 3])

    LL = L[:, 0::2, :, :]
    HL = L[:, 1::2, :, :]

    LL, HL = lifting97.lifting97_forward(LL, HL)

    LL = tf.transpose(LL, [0, 2, 1, 3])
    HL = tf.transpose(HL, [0, 2, 1, 3])

    # step 2: for w, H

    H = tf.transpose(H, [0, 2, 1, 3])

    LH = H[:, 0::2, :, :]
    HH = H[:, 1::2, :, :]

    LH, HH = lifting97.lifting97_forward(LH, HH)

    LH = tf.transpose(LH, [0, 2, 1, 3])
    HH = tf.transpose(HH, [0, 2, 1, 3])

    return LL, HL, LH, HH

############### one-level reconstruction of lifting 97

def reconstruct_fun(up,bot):

    temp_L = tf.transpose(up, [0, 2, 1, 3])
    temp_H = tf.transpose(bot, [0, 2, 1, 3])

    x_shape = temp_L.get_shape().as_list()

    x_n = x_shape[0]
    x_h = x_shape[1]
    x_w = x_shape[2]
    x_c = x_shape[3]

    temp_L = tf.reshape(temp_L, [x_n, x_h * x_w, 1, x_c])

    temp_H = tf.reshape(temp_H, [x_n, x_h * x_w, 1, x_c])

    temp = tf.concat([temp_L,temp_H],2)

    temp = tf.reshape(temp,[x_n, x_h, 2*x_w, x_c])

    recon = tf.transpose(temp, [0, 2, 1, 3])

    return recon

def reconstruct(LL,HL,LH,HH):

    LL = tf.transpose(LL, [0, 2, 1, 3])
    HL = tf.transpose(HL, [0, 2, 1, 3])

    LL, HL = lifting97.lifting97_inverse(LL, HL)

    L = reconstruct_fun(LL,HL)
    L = tf.transpose(L, [0, 2, 1, 3])



    LH = tf.transpose(LH, [0, 2, 1, 3])
    HH = tf.transpose(HH, [0, 2, 1, 3])

    LH, HH = lifting97.lifting97_inverse(LH, HH)

    H = reconstruct_fun(LH, HH)
    H = tf.transpose(H, [0, 2, 1, 3])



    L, H = lifting97.lifting97_inverse(L, H)

    recon = reconstruct_fun(L, H)


    return recon