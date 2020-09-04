from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from math import log
import numpy as np
from PIL import Image

import wavelet
import scale_quant
import entropy_codec
import creat_Long_context
import EDEH

decomposition_step = 4

def graph(x):

    LL = x

    # Convert RGB to YUV

    convert_mat = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.499], [0.499, -0.418, -0.0813]], dtype=np.float32)

    inverse_convert_mat = np.array([[1.0, 0.0, 1.402], [1.0, -0.344, -0.714], [1.0, 1.772, 0.0]], dtype=np.float32)

    Y = LL[:,:,:,0:1]    * convert_mat[0,0]  + LL[:,:,:,1:2]    * convert_mat[0,1]  + LL[:,:,:,2:3]    * convert_mat[0,2]
    U = LL[:, :, :, 0:1] * convert_mat[1, 0] + LL[:, :, :, 1:2] * convert_mat[1, 1] + LL[:, :, :, 2:3] * convert_mat[1, 2] + 128.
    V = LL[:, :, :, 0:1] * convert_mat[2, 0] + LL[:, :, :, 1:2] * convert_mat[2, 1] + LL[:, :, :, 2:3] * convert_mat[2, 2] + 128.

    # ########################### for Y

    LL = Y

    HL_collection = []
    LH_collection = []
    HH_collection = []

    # forward transform, bior4.4

    with tf.variable_scope('wavelet', reuse=tf.AUTO_REUSE):

        for i in range(decomposition_step):
            LL, HL, LH, HH = wavelet.decomposition(LL)

            HL_collection.append(HL)
            LH_collection.append(LH)
            HH_collection.append(HH)

    # quant, [x*QP]

    with tf.variable_scope('quant', reuse=tf.AUTO_REUSE):

        static_QP = tf.get_variable('static_QP', initializer=1 / 16)

    # static_QP = 1 / 39.2063

    LL, HL_collection, LH_collection, HH_collection = scale_quant.quant(LL, HL_collection, LH_collection, HH_collection,
                                                                        static_QP)

    with tf.variable_scope('long_context', reuse=tf.AUTO_REUSE):

        c_HL, c_LH, c_HH = creat_Long_context.context_all(LL, HL_collection, LH_collection, HH_collection, bit_map=1,
                                                          static_QP=static_QP * 4096.)

    with tf.variable_scope('ce_loss', reuse=tf.AUTO_REUSE):

        ce_loss_Y = entropy_codec.codec(LL, HL_collection, LH_collection, HH_collection, c_HL, c_LH, c_HH,
                                      static_QP=static_QP * 4096.)

    LL, HL_collection, LH_collection, HH_collection = scale_quant.de_quant(LL, HL_collection, LH_collection,
                                                                           HH_collection, static_QP)

    with tf.variable_scope('wavelet', reuse=tf.AUTO_REUSE):

        for j in range(decomposition_step):
            i = decomposition_step - 1 - j

            LL = wavelet.reconstruct(LL, HL_collection[i], LH_collection[i], HH_collection[i])

    LL = LL / 255. - 0.5

    with tf.variable_scope('post_process', reuse=tf.AUTO_REUSE):

        LL = EDEH.EDEH(LL)

    LL = (LL + 0.5) * 255.

    Y = LL

    # ##########################  for U

    LL = U

    HL_collection = []
    LH_collection = []
    HH_collection = []

    # forward transform, bior4.4

    with tf.variable_scope('wavelet', reuse=tf.AUTO_REUSE):

        for i in range(decomposition_step):
            LL, HL, LH, HH = wavelet.decomposition(LL)

            HL_collection.append(HL)
            LH_collection.append(LH)
            HH_collection.append(HH)

    # quant, [x*QP]

    with tf.variable_scope('quant', reuse=tf.AUTO_REUSE):

        static_QP = tf.get_variable('static_QP', initializer=1 / 16)

    # static_QP = 1 / 39.2063

    LL, HL_collection, LH_collection, HH_collection = scale_quant.quant(LL, HL_collection, LH_collection, HH_collection,
                                                                        static_QP)

    with tf.variable_scope('long_context', reuse=tf.AUTO_REUSE):

        c_HL, c_LH, c_HH = creat_Long_context.context_all(LL, HL_collection, LH_collection, HH_collection, bit_map=1,
                                                          static_QP=static_QP * 4096.)

    with tf.variable_scope('ce_loss', reuse=tf.AUTO_REUSE):

        ce_loss_U = entropy_codec.codec(LL, HL_collection, LH_collection, HH_collection, c_HL, c_LH, c_HH,
                                      static_QP=static_QP * 4096.)

    LL, HL_collection, LH_collection, HH_collection = scale_quant.de_quant(LL, HL_collection, LH_collection,
                                                                           HH_collection, static_QP)

    with tf.variable_scope('wavelet', reuse=tf.AUTO_REUSE):

        for j in range(decomposition_step):
            i = decomposition_step - 1 - j

            LL = wavelet.reconstruct(LL, HL_collection[i], LH_collection[i], HH_collection[i])

    LL = LL / 255. - 0.5

    with tf.variable_scope('post_process', reuse=tf.AUTO_REUSE):

        LL = EDEH.EDEH(LL)

    LL = (LL + 0.5) * 255.

    U = LL

    # ##########################  for V

    LL = V

    HL_collection = []
    LH_collection = []
    HH_collection = []

    # forward transform, bior4.4

    with tf.variable_scope('wavelet', reuse=tf.AUTO_REUSE):

        for i in range(decomposition_step):
            LL, HL, LH, HH = wavelet.decomposition(LL)

            HL_collection.append(HL)
            LH_collection.append(LH)
            HH_collection.append(HH)

    # quant, [x*QP]

    with tf.variable_scope('quant', reuse=tf.AUTO_REUSE):

        static_QP = tf.get_variable('static_QP', initializer=1 / 16)

    # static_QP = 1 / 39.2063

    LL, HL_collection, LH_collection, HH_collection = scale_quant.quant(LL, HL_collection, LH_collection, HH_collection,
                                                                        static_QP)

    with tf.variable_scope('long_context', reuse=tf.AUTO_REUSE):

        c_HL, c_LH, c_HH = creat_Long_context.context_all(LL, HL_collection, LH_collection, HH_collection, bit_map=1,
                                                          static_QP=static_QP * 4096.)

    with tf.variable_scope('ce_loss', reuse=tf.AUTO_REUSE):

        ce_loss_V = entropy_codec.codec(LL, HL_collection, LH_collection, HH_collection, c_HL, c_LH, c_HH,
                                      static_QP=static_QP * 4096.)

    LL, HL_collection, LH_collection, HH_collection = scale_quant.de_quant(LL, HL_collection, LH_collection,
                                                                           HH_collection, static_QP)

    with tf.variable_scope('wavelet', reuse=tf.AUTO_REUSE):

        for j in range(decomposition_step):
            i = decomposition_step - 1 - j

            LL = wavelet.reconstruct(LL, HL_collection[i], LH_collection[i], HH_collection[i])

    LL = LL / 255. - 0.5

    with tf.variable_scope('post_process', reuse=tf.AUTO_REUSE):

        LL = EDEH.EDEH(LL)

    LL = (LL + 0.5) * 255.

    V = LL


    # Convert YUV to RGB

    R = Y * inverse_convert_mat[0, 0] + (U - 128.) * inverse_convert_mat[0, 1] + (V - 128.) * inverse_convert_mat[0, 2]
    G = Y * inverse_convert_mat[1, 0] + (U - 128.) * inverse_convert_mat[1, 1] + (V - 128.) * inverse_convert_mat[1, 2]
    B = Y * inverse_convert_mat[2, 0] + (U - 128.) * inverse_convert_mat[2, 1] + (V - 128.) * inverse_convert_mat[2, 2]

    LL = tf.concat([R, G, B], axis=3)

    LL = tf.clip_by_value(LL, 0., 255.)



    d_loss = tf.losses.mean_squared_error(x, LL)

    ce_loss = ce_loss_Y + ce_loss_U + ce_loss_V

    return ce_loss/tf.log(2.), d_loss, ce_loss_Y/tf.log(2.), ce_loss_U/tf.log(2.), ce_loss_V/tf.log(2.), LL



h_in = 1200
w_in = 1200

x = tf.placeholder(tf.float32, [1, h_in, w_in, 3])

ce_loss, d_loss, ce_loss_Y, ce_loss_U, ce_loss_V, recon = graph(x)

saver = tf.train.Saver()

rate_all = []
psnr_all = []

with tf.Session() as sess:

    saver.restore(sess, 'your-model-path')

    for batch_index in range(100):

        i = batch_index + 1

        print('img_ID:', i)

        img = Image.open('Tecnick-path/RGB_OR_1200x1200_'+str(i).zfill(3)+'.png')

        img = np.asarray(img, dtype=np.float32)

        img = np.reshape(img, (1, h_in, w_in, 3))

        rate, d_eval, r_Y, r_U, r_V, recon_eval = sess.run([ce_loss, d_loss, ce_loss_Y, ce_loss_U, ce_loss_V, recon], feed_dict={x: img})

        # recon_eval = recon_eval[0, :, :, :]
        #
        # img = Image.fromarray(np.uint8(recon_eval))
        #
        # img.save('./recon/'+str(i)+'.png')

        psnr_ = 10 * log(255.*255. / d_eval, 10)

        print('PSNR:', psnr_)
        print('rate:', rate)

        rate_all.append(rate)
        psnr_all.append(psnr_)

    rate_all = np.array(rate_all)
    psnr_all = np.array(psnr_all)
    print('rate_mean:', np.mean(rate_all))
    print('psnr_mean:', np.mean(psnr_all))

    sess.close()
