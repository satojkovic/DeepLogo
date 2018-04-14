# The MIT License (MIT)
# Copyright (c) 2016 satojkovic

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import sys
import os
import common
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "train_dir", "flickr_logos_27_dataset",
    "Directory where to write event logs and checkpoint.")
tf.app.flags.DEFINE_integer("max_steps", 20001, "Number of batches to run.")
tf.app.flags.DEFINE_integer("image_width", common.CNN_IN_WIDTH,
                            "A width of an input image.")
tf.app.flags.DEFINE_integer("image_height", common.CNN_IN_HEIGHT,
                            "A height of an input image.")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "A batch size")
tf.app.flags.DEFINE_integer("num_channels", common.CNN_IN_CH,
                            "A number of channels of an input image.")

PICKLE_FILENAME = 'deep_logo.pickle'


def accuracy(predictions, labels):
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, FLAGS.image_height, FLAGS.image_width,
                               FLAGS.num_channels)).astype(np.float32)
    labels = (
        np.arange(model.NUM_CLASSES) == labels[:, None]).astype(np.float32)
    return dataset, labels


def read_data():
    with open(PICKLE_FILENAME, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Valid set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    return [train_dataset, valid_dataset,
            test_dataset], [train_labels, valid_labels, test_labels]


def main():
    if len(sys.argv) > 1:
        f = np.load(sys.argv[1])

        # f.files has unordered keys ['arr_8', 'arr_9', 'arr_6'...]
        # Sorting keys by value of numbers
        initial_weights = [
            f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))
        ]
    else:
        initial_weights = None

    # read input data
    dataset, labels = read_data()

    train_dataset, train_labels = reformat(dataset[0], labels[0])
    valid_dataset, valid_labels = reformat(dataset[1], labels[1])
    test_dataset, test_labels = reformat(dataset[2], labels[2])
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Valid set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    # Training model
    graph = tf.Graph()
    with graph.as_default():
        # Weights and biases
        model_params = model.params()

        # Initial weights
        if initial_weights is not None:
            assert len(model_params) == len(initial_weights)
            assign_ops = [
                w.assign(v) for w, v in zip(model_params, initial_weights)
            ]

        # Input data
        tf_train_dataset = tf.placeholder(
            tf.float32,
            shape=(FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width,
                   FLAGS.num_channels))
        tf_train_labels = tf.placeholder(
            tf.float32, shape=(FLAGS.batch_size, model.NUM_CLASSES))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Training computation
        logits = model.cnn(tf_train_dataset, model_params, keep_prob=0.5)
        with tf.name_scope('loss'):
            loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf_train_labels))
            tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            model.cnn(tf_valid_dataset, model_params, keep_prob=1.0))
        test_prediction = tf.nn.softmax(
            model.cnn(tf_test_dataset, model_params, keep_prob=1.0))
        # Merge all summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train')

        # Add ops to save and restore all the variables
        saver = tf.train.Saver()

    # Do training
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        if initial_weights is not None:
            session.run(assign_ops)
            print('initialized by pre-learned values')
        else:
            print('initialized')
        for step in range(FLAGS.max_steps):
            offset = (step * FLAGS.batch_size) % (
                train_labels.shape[0] - FLAGS.batch_size)
            batch_data = train_dataset[offset:(offset + FLAGS.batch_size
                                               ), :, :, :]
            batch_labels = train_labels[offset:(offset + FLAGS.batch_size), :]
            feed_dict = {
                tf_train_dataset: batch_data,
                tf_train_labels: batch_labels
            }
            try:
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
                if step % 50 == 0:
                    summary, _ = session.run(
                        [merged, optimizer], feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy(
                        predictions, batch_labels))
                    print('Validation accuracy: %.1f%%' % accuracy(
                        valid_prediction.eval(), valid_labels))
            except KeyboardInterrupt:
                last_weights = [p.eval() for p in model_params]
                np.savez("weights.npz", *last_weights)
                return last_weights

        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(),
                                                 test_labels))

        # Save the variables to disk.
        save_dir = "models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "deep_logo_model")
        saved = saver.save(session, save_path)
        print("Model saved in file: %s" % saved)


if __name__ == '__main__':
    main()
