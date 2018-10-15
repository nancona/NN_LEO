import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvas
from matplotlib.backends.backend_gtk3 import (
    NavigationToolbar2GTK3 as NavigationToolbar)

win = Gtk.Window()
win.connect("destroy", lambda x: Gtk.main_quit())
win.set_default_size(400, 300)
win.set_title("Embedding in GTK")

vbox = Gtk.VBox()
win.add(vbox)

# importing datasets from simulation
input_file = pd.read_csv("rbdl_leo2606_Animation-learn-0.csv")
input_file = input_file.values

training_epochs = 30000
display_step = 500
batch_size = 128
input_dim = 24
output_dim = 18
train_size = 0.8  # useless at the moment

sample_size = 10000

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

# neural network parameters
n_hidden_1 = 100
n_hidden_2 = 50
learning_rate = 0.001
Keep_prob = 0.8  # dropout probability

# save and restore ops


# Defining train input/output e test input/output
for i in range(sample_size):
    if i == 0:
        position_train = input_file[i][1:10]
        velocity_train = input_file[i][10:19]
        action_train = input_file[i][50:56]
        next_position_train = input_file[i][29:38]
        next_velocity_train = input_file[i][38:47]

    if i > 0 and i < 9000:
        position_train = np.vstack([position_train, input_file[i][1:10]])
        velocity_train = np.vstack([velocity_train, input_file[i][10:19]])
        action_train = np.vstack([action_train, input_file[i][50:56]])
        next_position_train = np.vstack([next_position_train, input_file[i][29:38]])
        next_velocity_train = np.vstack([next_velocity_train, input_file[i][38:47]])

    if i == 9000:
        position_test = input_file[i][1:10]
        velocity_test = input_file[i][10:19]
        action_test = input_file[i][50:56]
        next_position_test = input_file[i][29:38]
        next_velocity_test = input_file[i][38:47]

    if i > 9000:
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


# Comment/uncomment in order to define the multilayer net (1. one layer, 2. two layer)
# still in design --- smaller networks seem to perform better
def net(x, keep_prob):
    weights = {
        'h1': tf.Variable(tf.random_normal([input_dim, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, output_dim]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([output_dim]))
    }
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# def net(x, keep_prob):
#     weights = {
#         'h1': tf.Variable(tf.random_normal([input_dim, n_hidden_1])),
#         'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#         'out': tf.Variable(tf.random_normal([n_hidden_2, output_dim]))
#     }
#     biases = {
#         'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#         'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#         'out': tf.Variable(tf.random_normal([output_dim]))
#     }
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     layer_1 = tf.nn.dropout(layer_1, keep_prob)
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#     layer_2 = tf.nn.dropout(layer_2, keep_prob)
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     return out_layer


# weights = {
#     'h1': tf.Variable(tf.random_normal([input_dim, n_hidden_1])),
#     # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_1, output_dim]))
# }
#
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([output_dim]))
# }

keep_prob = tf.placeholder("float")
input = tf.placeholder("float", [None, input_dim])
output = tf.placeholder("float", [None, output_dim])

nn = net(x=input, keep_prob=keep_prob)
# nn = net(x=input, weights=weights, biases=biases, keep_prob=keep_prob)

cost = tf.losses.mean_squared_error(labels=output, predictions=nn)
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

    # saving model
    tf.train.Saver().save(sess, "./model.ckpt")

    test_input_pred = tf.convert_to_tensor(test_input)
    prediction = nn.eval({input: test_input, keep_prob: 1})
    print prediction
    print test_output

    # plot predictions
    time_sample = 0.03
    sample_plot = np.arange(0, 100, 1)
    time = sample_plot * time_sample

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(time, test_output[0:100, 2], 'g', time, prediction[0:100, 2], 'b--')

    canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
    vbox.pack_start(canvas, True, True, 0)
    toolbar = NavigationToolbar(canvas, win)
    vbox.pack_start(toolbar, False, False, 0)

    win.show_all()
    Gtk.main()
    # plt.plot(time, test_output[:,2],'g', time, prediction[:,2], 'b--')
    # plt.show()

    # # print sess.run(nn, feed_dict={input: test_input})

    # evaluation of the model (still not clear)
    correct_prediction = tf.equal(tf.argmax(nn, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({input: test_input, output: test_output, keep_prob: 1.0}))


# with tf.Session() as sess:
#     ckpt = tf.train.get_checkpoint_state('./model')
#     tf.train.Saver().restore(sess, ckpt.model_checkpoint)
#     #feed_dict = {input: test_input}
#     predictions = sess.run([test_input])


