#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import rnn, rnn_cell
from util import download, read_data

import logging


def main(argv):
    url = 'http://mattmahoney.net/dc/'
    filename = download(url, 'text8.zip')

    text = read_data(filename)
    vocabulary = set(text)
    vocabulary_size = len(vocabulary)

    # Parameters
    learning_rate = 0.1
    training_iters = 1000

    # Network Parameters
    n_input = len(vocabulary)
    n_steps = 10
    n_hidden = 128
    n_classes = len(vocabulary)

    x = tf.placeholder("int32", [None, n_steps])

    output_layer = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    word_to_index = {word:idx for idx, word in enumerate(vocabulary)}

    def RNN(x, output_layer):
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(0, n_steps, x)
        lstm_cell = rnn_cell.BasicRNNCell(n_hidden)
        rnn_outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return [tf.nn.softmax(tf.matmul(ro, output_layer)) for ro in rnn_outputs]

    one_hot_vectors = tf.one_hot(x, n_input)
    model = RNN(one_hot_vectors, output_layer)

    # Define loss and optimizer

    true_output = one_hot_vectors[:, 1:]
    predicted_output = model[:-1]

    cross_entropy = - tf.reduce_sum(true_output * tf.squeeze(tf.log(predicted_output)), reduction_indices=[0, 1, 2])

    cost = cross_entropy
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    random_state = np.random.RandomState(0)

    # Launch the graph
    with tf.Session() as session:
        session.run(init)
        step = 1

        # Keep training until reach max iterations
        while step < training_iters:
            start_idx = random_state.randint(vocabulary_size - n_steps)
            end_idx = start_idx + n_steps

            sub_text = [text[idx] for idx in range(start_idx, end_idx)]
            sub_text_idxs = [word_to_index[word] for word in sub_text]

            session.run(cost, feed_dict={x: [sub_text_idxs]})
            loss = session.run(cost, feed_dict={x: [sub_text_idxs]})

            logging.info("Iter {}, Minibatch Loss= {:.6f}".format(step, loss))
            step += 1

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
