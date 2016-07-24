#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
from skimage.io import imread

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
tf.app.flags.DEFINE_integer("learning_rate", 0.01, "Learning rate")
tf.app.flags.DEFINE_integer("batch_size", 16, "A batch size")
tf.app.flags.DEFINE_integer("num_channels", 3, "A number of channels of an input image.")


def train():
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
                                                             FLAGS.image_width,
                                                             FLAGS.image_height,
                                                             FLAGS.num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
                                                            FLAGS.num_classes))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)


def main():
    if not tf.gfile.Exists(FLAGS.train_dir):
        print("Not found: %s" % (FLAGS.train_dir))
    train()


if __name__ == '__main__':
    main()
