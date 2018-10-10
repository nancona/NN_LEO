import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets from simulation
input_file = pd.read_csv("rbdl_leo2606_Animation-learn-0.csv")
input_file = input_file.values

training_epochs = 5000
display_step = 1000
batch_size = 64
input_dim = 24
output_dim = 18
train_size = 0.8

# TO DO: define train input e test input
train_input = 0

def net (x, weights, biases, keep_prob):

    layer_1 = tf.add(tf.matmul(x, weights['h1'], biases['b1']))
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer
n_hidden_1 = 300

n_input = train

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, output_dim]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([output_dim]))
}

keep_prob = tf.placeholder("float")
input = tf.placeholder("float", [None, input_dim])
output = tf.placeholder("float", [None, output_dim])

nn = net(x=input, weights=weights, biases=biases, keep_prob=keep_prob)

cost = tf.losses.mean_squared_error(labels=output, predictions=nn)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(input_train/batch_size))
        input_train_batches = np.array_split(input_train, total_batch)
        output_train_batches = np.array_split(output_train, total_batch)

        for i in range(total_batch):
            batch_it, batch_ot = input_train_batches[i], output_train_batches[i]
            _, c = sess.run([optimizer, cost],
                            feed_dict={
                                input:batch_it,
                                output:batch_ot,
                                keep_prob:1})
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print "Optimization Finished!"

    correct_prediction = tf.equal(tf.argmax(nn, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({input: input_test, output: output_test, keep_prob: 1.0}))
