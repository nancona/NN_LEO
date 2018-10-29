import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import os
import datetime

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_gtk3agg import FigureCanvas
from matplotlib.backends.backend_gtk3 import (
    NavigationToolbar2GTK3 as NavigationToolbar)

from net import net

win = Gtk.Window()
win.connect("destroy", lambda x: Gtk.main_quit())
win.set_default_size(400, 300)
win.set_title("Embedding in GTK")

vbox = Gtk.VBox()
win.add(vbox)

# importing datasets from simulation
input_file = pd.read_csv("rbdl_leo2606_Animation-learn-0.csv")
input_file = input_file.values

training_epochs = 20000
display_step = 100
batch_size = 64
input_dim = 24
output_dim = 18
train_size = 1.0  # useless at the moment
Keep_prob = 1.0  # dropout probability

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

# save and restore model
# change the name if settings are different
# name meening ===> model_{#epochs}_{keep_prob}
model_name = 'model_%sk_%s.ckpt' % (int(training_epochs/1000), int(Keep_prob*100))
settings_folder = './results_model_nn/%sk_%s' % (int(training_epochs/1000), int(Keep_prob*100))
now = datetime.datetime.now()
date_folder = '%s_%s_%s_%s_%s' % (now.day, now.month, now.year, now.hour, now.minute)
model_path = "model_restore"
results_folder = os.path.join(settings_folder, date_folder)
model_folder = os.path.join(settings_folder, date_folder, model_path)

if not os.path.exists(settings_folder):
    os.makedirs(settings_folder)
os.makedirs(results_folder)
os.makedirs(model_folder)
# Defining train input/output e test input/output
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

# mixing data
permutation = np.random.permutation(train_input.shape[0])
train_input = train_input[permutation]
train_output = train_output[permutation]


# ax.plot(time, prediction[0:100, 2], 'b--', label='prediction')
# Comment/uncomment in order to define the multilayer net (1. one layer, 2. two layer)
# still in design --- smaller networks seem to perform better

keep_prob = tf.placeholder("float")
input = tf.placeholder("float", [None, input_dim])
output = tf.placeholder("float", [None, output_dim])

nn = net(x=input, keep_prob=keep_prob)
# nn = net(x=input, weights=weights, biases=biases, keep_prob=keep_prob)

cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=output, predictions=nn))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(train_input) / batch_size)
        train_input_batches = np.array_split(train_input, total_batch)
        train_output_batches = np.array_split(train_output, total_batch)

        for i in range(total_batch):
            batch_it, batch_ot = train_input_batches[i], train_output_batches[i]
            _, c = sess.run([optimizer, cost],
                            feed_dict={
                                input: batch_it,
                                output: batch_ot,
                                keep_prob: Keep_prob})
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print "Optimization Finished!"

    # evaluation of the model (still not clear)
    correct_prediction = tf.equal(tf.argmax(nn, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({input: test_input, output: test_output, keep_prob: 1.0}))


    # saving model
    model_folder = os.path.join(model_folder, model_name)
    tf.train.Saver().save(sess, model_folder)

    # test_input_pred = tf.convert_to_tensor(test_input)
    prediction = nn.eval({input: test_input, keep_prob: 1})
    # print prediction
    # print test_output

    # plot predictions
    time_sample = 0.03
    sample_plot = np.arange(0, 100, 1)
    time = sample_plot * time_sample

    prediction_path = os.path.join(results_folder, "prediction.txt")
    validation_path = os.path.join(results_folder, "test_dataset.txt")
    np.savetxt(prediction_path, prediction, delimiter='\t')
    np.savetxt(validation_path, test_output, delimiter='\t')

    # rc('text', usetex=True)
    # plt.figure(num=1, figsize=(5, 4), dpi=100)
    # plt.plot(time, prediction[0:100, 12], label='prediction')
    # plt.plot(time, test_output[0:100, 12], label='validation')
    # plt.title('Left Hip Angular Velocity')
    # plt.xlabel('time [s]')
    # plt.ylabel('dotalpha [rad*$s^-1$]')
    # plt.legend(loc='upper left')
    #
    # plt.show()
    #
    # fig = plt.figure(figsize=(5, 4), dpi=100)
    # ax = fig.add_subplot(111)
    # ax.plot(time, test_output[0:100, 2], 'g', label='validation')
    # ax.plot(time, prediction[0:100, 2], 'b--', label='prediction')
    # ax.set_title('plot_title')
    # ax.set_xlabel('time [s]')
    # ax.set_ylabel('position [m]')
    # ax.legend()
    #
    # canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
    # vbox.pack_start(canvas, True, True, 0)
    # toolbar = NavigationToolbar(canvas, win)
    # vbox.pack_start(toolbar, False, False, 0)

    rmse = np.zeros(18)
    # RMSE
    for k in range(0, 18):
        rmse[k] = sqrt(mean_squared_error(test_output[:, k], prediction[:, k]))
        print "RMSE", '%04d' % (k), rmse[k]
    rmse_path = os.path.join(results_folder, "rmse.txt")
    np.savetxt(rmse_path, rmse, delimiter='\n')

    win.show_all()
    Gtk.main()

