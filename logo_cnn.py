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


def inference():
    pass


def train():
    with tf.Graph().as_default():
        images, labels = cropped_inputs()

        logits = inference()


def main():
    if not tf.gfile.Exists(FLAGS.train_dir):
        print("Not found: %s" % (FLAGS.train_dir))
    read_flickrlogos27(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    main()
