#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import logging

def main(argv):
    lstm = rnn_cell.BasicLSTMCell(lstm_size)
    # Initial state of the LSTM memory.
    state = tf.zeros([batch_size, lstm.state_size])
    probabilities = []
    loss = 0.0
    for current_batch_of_words in words_in_dataset:
        # The value of state is updated after processing each batch of words.
        output, state = lstm(current_batch_of_words, state)
        # The LSTM output can be used to make next word predictions
        logits = tf.matmul(output, softmax_w) + softmax_b
        probabilities.append(tf.nn.softmax(logits))
        loss += loss_function(probabilities, target_words)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
