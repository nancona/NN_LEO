import tensorflow as tf
import os
import pandas as pd
import numpy as np
from net import leo_nn
from sklearn.metrics import mean_squared_error
from math import sqrt

# ==============================================
# Restore session
# ==============================================
input_dim = 24
output_dim = 18
Keep_prob = 1.0

# imported_meta = tf.train.import_meta_graph(meta_file)



sample_size = 8000

# Sample vector initialization
position_train = []
velocity_train = []
action_train = []
next_position_train = []
next_velocity_train = []

position_test = []
velocity_test = []
action_test = []
next_position_test = []
next_velocity_test = []

input_file = pd.read_csv("rbdl_leo2606_Animation-learn-0.csv")
input_file = input_file.values

for i in range(sample_size):
    if i == 0:
        position_train = input_file[i][1:10]
        velocity_train = input_file[i][10:19]
        action_train = input_file[i][50:56]
        next_position_train = input_file[i][29:38]
        next_velocity_train = input_file[i][38:47]

    if i > 0 and i < 5000:
        position_train = np.vstack([position_train, input_file[i][1:10]])
        velocity_train = np.vstack([velocity_train, input_file[i][10:19]])
        action_train = np.vstack([action_train, input_file[i][50:56]])
        next_position_train = np.vstack([next_position_train, input_file[i][29:38]])
        next_velocity_train = np.vstack([next_velocity_train, input_file[i][38:47]])

    if i == 7000:
        position_test = input_file[i][1:10]
        velocity_test = input_file[i][10:19]
        action_test = input_file[i][50:56]
        next_position_test = input_file[i][29:38]
        next_velocity_test = input_file[i][38:47]

    if i > 7000:
        position_test = np.vstack([position_test, input_file[i][1:10]])
        velocity_test = np.vstack([velocity_test, input_file[i][10:19]])
        action_test = np.vstack([action_test, input_file[i][50:56]])
        next_position_test = np.vstack([next_position_test, input_file[i][29:38]])
        next_velocity_test = np.vstack([next_velocity_test, input_file[i][38:47]])

# Train samples vector
train_input = np.hstack([position_train, velocity_train, action_train])
train_output = np.hstack([next_position_train, next_velocity_train])
# Test samples vector
state_input = np.hstack([position_test, velocity_test, action_test])
test_output = np.hstack([next_position_test, next_velocity_test])

def load_model(load=True):
    # tf.reset_default_graph()
    #
    # keep_prob = tf.placeholder("float")
    # input = tf.placeholder("float", [None, input_dim])
    # output = tf.placeholder("float", [None, output_dim])
    # nn = net(x=input, keep_prob=keep_prob)

    # with tf.Session() as sess:
    # def restore(sess):
    #     tf.train.Saver().restore(sess,
    #         '/home/nancona/PycharmProjects/NN_LEO/results_model_nn/20k_100/29_10_2018_10_7/model_restore/model_20k_100.ckpt')

    model = leo_nn()
    prediction = model.restore()

    prediction = model.eval({input: state_input, keep_prob: 1})
    print prediction
    #     # r
    # sess = tf.Session()
    # # if load:
    # tf.train.Saver().restore(sess,
    #                              '/home/nancona/PycharmProjects/NN_LEO/results_model_nn/20k_100/29_10_2018_10_7/model_restore/model_20k_100.ckpt')
    #
    #         # print tf.get_default_graph().as_graph_def()
    # return nn,sess

# load_model()