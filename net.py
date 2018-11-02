import tensorflow as tf

# neural network parameters
n_hidden_1 = 200
n_hidden_2 = 200
n_hidden_3 = 200

input_dim = 24
output_dim = 18
Keep_prob = 1.0

class leo_nn:

    def __init__(self, sess, n_hidden_1=200, input_dim=24, output_dim=18, Keep_prob=1.0):
        self.sess = sess
        self.n_hidden_1 = n_hidden_1
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Keep_prob = Keep_prob
        self.keep_prob = tf.placeholder("float")
        self.input = tf.placeholder("float", [None, input_dim])
        self.output = tf.placeholder("float", [None, output_dim])
        self.model_path = '/home/nancona/PycharmProjects/NN_LEO/results_model_nn/20k_100/29_10_2018_10_7/model_restore/model_20k_100.ckpt'
        self.initialize = self.net(x=self.input)



    def net(self, x):
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.input_dim, self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, self.output_dim]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.output_dim]))
        }
        self.layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        self.layer_1 = tf.nn.relu(self.layer_1)
        self.layer_1 = tf.nn.dropout(self.layer_1, self.Keep_prob)
        self.out_layer = tf.matmul(self.layer_1, self.weights['out']) + self.biases['out']

        return self.out_layer

    def restore(self):
        # saver = tf.train.import_meta_graph(
        #     self.model_path + '.meta', clear_devices=True)
        # with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def prediction(self, measured_input):
        prediction = self.sess.run(self.initialize, feed_dict={self.input: measured_input})
        return prediction




# def net2(x, keep_prob):
#     weights = {
#         'h1': tf.Variable(tf.random_normal([input_dim, n_hidden_1])),
#         'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#         'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
#         'out': tf.Variable(tf.random_normal([n_hidden_3, output_dim]))
#     }
#     biases = {
#         'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#         'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#         'b3': tf.Variable(tf.random_normal([n_hidden_3])),
#         'out': tf.Variable(tf.random_normal([output_dim]))
#     }
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     layer_1 = tf.nn.dropout(layer_1, keep_prob)
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#     layer_2 = tf.nn.dropout(layer_2, keep_prob)
#     layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
#     layer_3 = tf.nn.relu(layer_3)
#     layer_3 = tf.nn.dropout(layer_3, keep_prob)
#     out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
#     return out_layer