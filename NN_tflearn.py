import tensorflow as tf
import tflearn
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
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

training_epochs = 10000
display_step = 500
batch_size = 128
input_dim = 24
output_dim = 18
train_size = 0.8  # useless at the moment

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

# neural network parameters
n_hidden_1 = 200
n_hidden_2 = 200
n_hidden_3 = 200
learning_rate = 0.001
keep_prob = 0.8  # dropout probability

# save and restore ops


# Defining train input/output e test input/output
for i in range(sample_size):
    if i == 0:
        position_train = input_file[i][1:10]
        velocity_train = input_file[i][10:19]
        action_train = input_file[i][50:56]
        next_position_train = input_file[i][29:38]
        next_velocity_train = input_file[i][38:47]

    if i > 0 and i < 7000:
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



# Neural Network
def create_actor_network():
    inputs = tflearn.input_data(shape=[None, input_dim])
    layer1 = tflearn.fully_connected(inputs, 300, activation='relu', name="leoLayer1",
                                           weights_init=tflearn.initializations.uniform(
                                               minval=-1, maxval=1))
    dropout1 = tflearn.dropout(layer1, keep_prob=keep_prob)
    layer2 = tflearn.fully_connected(dropout1, 300, activation='relu', name="leoLayer2",
                                           weights_init=tflearn.initializations.uniform(minval=-1,
                                                                                        maxval=1))
    dropout2 = tflearn.dropout(layer2, keep_prob=keep_prob)
    w_init = tflearn.initializations.uniform(minval=-1, maxval=1)
    layer_output = tflearn.fully_connected(dropout2, output_dim, activation='relu', weights_init=w_init,
                                           name="leoOutput")
    return layer_output


nn = tflearn.regression(create_actor_network(), optimizer="adam", loss="mean_square",
                        metric="R2", learning_rate=learning_rate, batch_size=batch_size)

model = tflearn.DNN(nn)
model.fit(train_input, train_output, n_epoch=training_epochs, show_metric=False)

print("Final Accurancy:", model.evaluate(test_input, test_output))