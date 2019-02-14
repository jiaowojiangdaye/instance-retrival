#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 22:21:44 2019

@author: aaron
"""

import tensorflow as tf
import numpy as np
import keras.layers as KL
a = np.array([[[0,0,0,0,0]], [[1,1,1,1,1]], [[2,2,2,2,2]], [[3,3,3,3,3]]])

#input_a = tf.placeholder(tf.int16, shape=(4,1,5), name='input_a')

input_a = tf.constant(a, dtype=tf.int32)

input_a = tf.squeeze(input_a)

split = tf.dynamic_partition(input_a, [0,0, 1,1],2)
[split0, split1] = split
split11 = tf.transpose(split1)
re= tf.matmul(split0, split11)

tf.initialize_all_variables()
sess = tf.Session()


out_put, re = sess.run([split, re])




print(out_put)

