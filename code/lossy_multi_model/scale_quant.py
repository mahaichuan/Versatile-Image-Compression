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

clip_value = 8192.

def quant(LL, HL_collection, LH_collection, HH_collection, s):

    with tf.get_default_graph().gradient_override_map({"Round": "Identity"}):

        HL_collection[0] = tf.round( s * tf.clip_by_value(HL_collection[0], -clip_value, clip_value) )
        LH_collection[0] = tf.round( s * tf.clip_by_value(LH_collection[0], -clip_value, clip_value) )
        HH_collection[0] = tf.round( s * tf.clip_by_value(HH_collection[0], -clip_value, clip_value) )

        HL_collection[1] = tf.round( s * tf.clip_by_value(HL_collection[1], -clip_value, clip_value) )
        LH_collection[1] = tf.round( s * tf.clip_by_value(LH_collection[1], -clip_value, clip_value) )
        HH_collection[1] = tf.round( s * tf.clip_by_value(HH_collection[1], -clip_value, clip_value) )

        HL_collection[2] = tf.round( s * tf.clip_by_value(HL_collection[2], -clip_value, clip_value) )
        LH_collection[2] = tf.round( s * tf.clip_by_value(LH_collection[2], -clip_value, clip_value) )
        HH_collection[2] = tf.round( s * tf.clip_by_value(HH_collection[2], -clip_value, clip_value) )

        HL_collection[3] = tf.round( s * tf.clip_by_value(HL_collection[3], -clip_value, clip_value) )
        LH_collection[3] = tf.round( s * tf.clip_by_value(LH_collection[3], -clip_value, clip_value) )
        HH_collection[3] = tf.round( s * tf.clip_by_value(HH_collection[3], -clip_value, clip_value) )

        LL = tf.round( s * tf.clip_by_value(LL, -clip_value, clip_value) )


    return LL, HL_collection, LH_collection, HH_collection


def de_quant(LL, HL_collection, LH_collection, HH_collection, s):

    HL_collection[0] = HL_collection[0] / s
    LH_collection[0] = LH_collection[0] / s
    HH_collection[0] = HH_collection[0] / s

    HL_collection[1] = HL_collection[1] / s
    LH_collection[1] = LH_collection[1] / s
    HH_collection[1] = HH_collection[1] / s

    HL_collection[2] = HL_collection[2] / s
    LH_collection[2] = LH_collection[2] / s
    HH_collection[2] = HH_collection[2] / s

    HL_collection[3] = HL_collection[3] / s
    LH_collection[3] = LH_collection[3] / s
    HH_collection[3] = HH_collection[3] / s

    LL = LL / s

    return LL, HL_collection, LH_collection, HH_collection