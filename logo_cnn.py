#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "train_dir", "flickr_logos_27_dataset",
    "Directory where to write event logs and checkpoint.")
tf.app.flags.DEFINE_integer("max_steps", 10000, "Number of batches to run.")
tf.app.flags.DEFINE_integer("image_size", 64, "Size of an input image.")
tf.app.flags.DEFINE_integer("num_classes", 27, "Number of logo classes.")
tf.app.flags.DEFINE_integer("learning_rate", 0.01, "Learning rate")


def read_flickrlogos27(filename_queue):
    class FlickrLogos27Record():
        pass

    result = FlickrLogos27Record()

    label_bytes = 1
    result.width = 32
    result.height = 32
    result.depth = 3
    image_bytes = result.width * result.height * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)


def cropped_inputs():
    if tf.gfile.Exists(os.path.join(
            FLAGS.train_dir,
            'flickr_logos_27_dataset_training_set_annotation.txt')):
        raise ValueError(
            'Failed to find file: flickr_logos_27_dataset_training_set_annotation.txt')
    annot_train = np.loadtxt(
        os.path.join(FLAGS.train_dir,
                     'flickr_logos_27_dataset_training_set_annotation.txt'))

    filenames = [os.path.join(FLAGS.train_dir, annot[0])
                 for annot in annot_train]

    filename_queue = tf.train.input_producer(filenames)

    read_input = read_flickrlogos27(filename_queue)


def convolutional_layer():
    x = tf.placeholder(tf.float32, [None, None, None])

    # first layer
    w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 48], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, [48]))
    x_expanded = tf.expand_dims(x, 3)
    conv1 = tf.nn.conv2d(x_expanded, w_conv1, strides=(1, 1), padding='SAME')
    h_conv1 = tf.nn.relu(conv1 + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1,
                             ksize=[1, 2, 2, 1],
                             stride=[1, 2, 2, 1],
                             padding='SAME')

    # second layer
    w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 48, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, [64]))
    conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=(1, 1), padding='SAME')
    h_conv2 = tf.nn.relu(conv2 + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,
                             ksize=[1, 2, 1, 1],
                             stride=[1, 2, 1, 1],
                             padding='SAME')

    # third layer
    w_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1))
    b_conv3 = tf.Variable(tf.constant(0.1, [128]))
    conv3 = tf.nn.conv2d(h_pool2, w_conv3, strides=(1, 1), padding='SAME')
    h_conv3 = tf.nn.relu(conv3 + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

    return x, h_pool3, [w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3]


def inference():
    x, conv_layer, conv_vars = convolutional_layer()

    # Densely connected layer
    w_fc1 = tf.Variable(tf.tuncated_normal([32 * 8 * 128, 2048]))
    b_fc1 = tf.Variable(tf.constant(0.1, [2048]))
    conv_layer_flat = tf.reshape(conv_layer, [-1, 32 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, w_fc1) + b_fc1)

    # Output layer
    w_fc2 = tf.Variable(tf.truncated_normal([2048, FLAGS.num_classes]))
    b_fc2 = tf.Variable(tf.constant(0.1, [FLAGS.num_classes]))

    y = tf.matmul(h_fc1, w_fc2) + b_fc2

    return (x, y, conv_vars + [w_fc1, b_fc1, w_fc2, b_fc2])


def train():
    x, y, params = inference()

    y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    logo_loss = tf.nn.softmax_cross_entropy_with_logits(
        tf.reshape(y[:, 1:], [-1, FLAGS.num_classes]),
        tf.reshape(y_[:, 1:], [-1, FLAGS.num_classes]))

    logo_loss = tf.reduce_sum(logo_loss)

    train_step = tf.train.AdamOptimizer(FLAGS.learn_rate).minimize(logo_loss)

    best = tf.argmax(tf.reshape(y[:, 1:], [-1, FLAGS.num_classes]), 1)
    correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, FLAGS.num_classes]), 1)

    init = tf.initialize_all_variables()

        logo_loss = tf.reduce_sum(logo_loss)


def main():
    if not tf.gfile.Exists(FLAGS.train_dir):
        print("Not found: %s" % (FLAGS.train_dir))
    train()


if __name__ == '__main__':
    main()
