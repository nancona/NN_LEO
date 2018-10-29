import tensorflow as tf
import os
import pandas as pd
import numpy as np
from net import net
from sklearn.metrics import mean_squared_error
from math import sqrt

# ==============================================
# Restore session
# ==============================================
input_dim = 24
output_dim = 18
Keep_prob = 1.0

def restore():
    # imported_meta = tf.train.import_meta_graph(meta_file)
    keep_prob = tf.placeholder("float")
    input = tf.placeholder("float", [None, input_dim])
    output = tf.placeholder("float", [None, output_dim])
    nn = net(x=input, keep_prob=keep_prob)

    # with tf.Session() as sess:
    # def restore(sess):
    #     tf.train.Saver().restore(sess,
    #         '/home/nancona/PycharmProjects/NN_LEO/results_model_nn/20k_100/29_10_2018_10_7/model_restore/model_20k_100.ckpt')
    #     # prediction = nn.eval({input: state_input, keep_prob: 1})
    #     # print prediction
    #     # return prediction

    with tf.Session() as sess:
        tf.train.Saver().restore(sess,
            '/home/nancona/PycharmProjects/NN_LEO/results_model_nn/20k_100/29_10_2018_10_7/model_restore/model_20k_100.ckpt')
