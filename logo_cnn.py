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
tf.app.flags.DEFINE_integer("learning_rate", 0.01, "Learning rate")
tf.app.flags.DEFINE_integer("batch_size", 16, "A batch size")
tf.app.flags.DEFINE_integer("num_channels", 3,
                            "A number of channels of an input image.")

PICKLE_FILENAME = 'deep_logo.pickle'


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


if __name__ == '__main__':
    main()
