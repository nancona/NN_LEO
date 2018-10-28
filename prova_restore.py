import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# ==============================================
# Restore session
# ==============================================
input_file = pd.read_csv("rbdl_leo2606_Animation-learn-0.csv")
input_file = input_file.values

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

training_epochs = 100
display_step = 100
batch_size = 64
input_dim = 24
output_dim = 18
train_size = 0.8  # useless at the moment

sample_size = 8000
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
test_input = np.hstack([position_test, velocity_test, action_test])
test_output = np.hstack([next_position_test, next_velocity_test])

with tf.Session() as sess:
    model_path = \
        os.path.join('/home/nicola/PycharmProjects/NN_LEO/results_model_nn/0k_80/28_10_2018_17_50/model_restore/model_0k_80.ckpt')
    tf.train.Saver().restore(sess, model_path)
    prediction = sess.eval({input: test_input})

    # RMSE restore to check if it works!
    for i in range(0, 18):
        rmse = sqrt(mean_squared_error(test_output[:, i], prediction[:, i]))
        print "RMSE", '%04d' % (i), rmse