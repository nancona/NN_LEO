import tensorflow as tf
import tflearn
import math

keep_prob = 0.8

class LEO_NN(object):

    def __init__(self, sess, i_dim, o_dim, learning_rate, tau):
        self.sess = sess
        self.i_dim = i_dim
        self.o_dim = o_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)


    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.i_dim])
        layer1 = tflearn.fully_connected(inputs, 300, activation='relu', name="leoLayer1",
                                               weights_init=tflearn.initializations.uniform(
                                                   minval=-1, maxval=1))
        dropout1 = tflearn.dropout(layer1, keep_prob=keep_prob)
        layer2 = tflearn.fully_connected(dropout1, 300, activation='relu', name="leoLayer2",
                                               weights_init=tflearn.initializations.uniform(minval=-1,
                                                                                            maxval=1))
        dropout2 = tflearn.dropout(layer2, keep_prob=keep_prob)
        w_init = tflearn.initializations.uniform(minval=-1, maxval=1)
        layer_output = tflearn.fully_connected(dropout2, self.o_dim, activation='relu', weights_init=w_init,
                                               name="actorOutput")
        return layer_output


    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars