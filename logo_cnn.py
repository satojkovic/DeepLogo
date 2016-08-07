#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range

CLASS_NAMES = ['Adidas'
               'Apple'
               'BMW'
               'Citroen'
               'Cocacola'
               'DHL'
               'Fedex'
               'Ferrari'
               'Ford'
               'Google'
               'Heineken'
               'HP'
               'Intel'
               'McDonalds'
               'Mini'
               'Nbc'
               'Nike'
               'Pepsi'
               'Porsche'
               'Puma'
               'RedBull'
               'Sprite'
               'Starbucks'
               'Texaco'
               'Unicef'
               'Vodafone'
               'Yahoo']

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "train_dir", "flickr_logos_27_dataset",
    "Directory where to write event logs and checkpoint.")
tf.app.flags.DEFINE_integer("max_steps", 10001, "Number of batches to run.")
tf.app.flags.DEFINE_integer("image_width", 64, "A width of an input image.")
tf.app.flags.DEFINE_integer("image_height", 32, "A height of an input image.")
tf.app.flags.DEFINE_integer("num_classes", 27, "Number of logo classes.")
tf.app.flags.DEFINE_integer("learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_integer("batch_size", 16, "A batch size")
tf.app.flags.DEFINE_integer("num_channels", 3,
                            "A number of channels of an input image.")
tf.app.flags.DEFINE_integer("patch_size", 5,
                            "A patch size of convolution filter")

PICKLE_FILENAME = 'deep_logo.pickle'


def accuracy(predictions, labels):
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, FLAGS.image_width, FLAGS.image_height,
                               FLAGS.num_channels)).astype(np.float32)
    labels = (
        np.arange(FLAGS.num_classes) == labels[:, None]).astype(np.float32)
    return dataset, labels


def model(data):
    # First layer
    w_conv1 = tf.Variable(
        tf.truncated_normal(
            [FLAGS.patch_size, FLAGS.patch_size, FLAGS.num_channels, 48],
            stddev=0.1))
    b_conv1 = tf.constant(0.1, shape=[48])
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(
            data, w_conv1, [1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(
        h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second layer
    w_conv2 = tf.Variable(
        tf.truncated_normal(
            [FLAGS.patch_size, FLAGS.patch_size, 48, 64], stddev=0.1))
    b_conv2 = tf.constant(0.1, shape=[64])
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(
            h_pool1, w_conv2, [1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(
        h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    # Third layer
    w_conv3 = tf.Variable(
        tf.truncated_normal(
            [FLAGS.patch_size, FLAGS.patch_size, 64, 128], stddev=0.1))
    b_conv3 = tf.constant(0.1, shape=[128])
    h_conv3 = tf.nn.relu(
        tf.nn.conv2d(
            h_pool2, w_conv3, [1, 1, 1, 1], padding='SAME') + b_conv3)
    h_pool3 = tf.nn.max_pool(
        h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    w_fc1 = tf.Variable(tf.truncated_normal([16 * 4 * 128, 2048], stddev=0.1))
    b_fc1 = tf.constant(0.1, shape=[2048])
    conv_layer_flat = tf.reshape(h_pool3, [-1, 16 * 4 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, w_fc1) + b_fc1)

    # Output layer
    w_fc2 = tf.Variable(tf.truncated_normal([2048, FLAGS.num_classes]))
    b_fc2 = tf.constant(0.1, shape=[FLAGS.num_classes])
    out = tf.matmul(h_fc1, w_fc2) + b_fc2

    return out


def main():
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

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Valid set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    # Training model
    graph = tf.Graph()
    with graph.as_default():
        # Input data
        tf_train_dataset = tf.placeholder(
            tf.float32,
            shape=(FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height,
                   FLAGS.num_channels))
        tf_train_labels = tf.placeholder(
            tf.float32, shape=(FLAGS.batch_size, FLAGS.num_classes))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Training computation
        logits = model(tf_train_dataset)
        loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

    # Do training
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('initialized')
        for step in range(FLAGS.max_steps):
            offset = (step * FLAGS.batch_size) % (
                train_labels.shape[0] - FLAGS.batch_size)
            batch_data = train_dataset[offset:(offset + FLAGS.batch_size
                                               ), :, :, :]
            batch_labels = train_labels[offset:(offset + FLAGS.batch_size), :]
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 50 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions,
                                                              batch_labels))
                print('Validation accuracy: %.1f%%' %
                      accuracy(valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(),
                                                 test_labels))


if __name__ == '__main__':
    main()
