#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "train_dir", "logo_images/cropped",
    "Directory where to write event logs and checkpoint.")
tf.app.flags.DEFINE_integer("max_steps", 10000, "Number of batches to run.")
tf.app.flags.DEFINE_integer("image_size", 64, "Size of an input image.")


def read_flickrlogos27(train_dir):
    class FlickrLogos27Record():
        pass

    result = FlickrLogos27Record()

    label_bytes = 1
    result.width = 32
    result.height = 32
    result.depth = 3
    image_bytes = result.width * result.height * result.depth
    record_bytes = label_bytes + image_bytes



def inference():
    pass


def train():
    with tf.Graph().as_default():
        logits = inference()


def main():
    if not tf.gfile.Exists(FLAGS.train_dir):
        print("Not found: %s" % (FLAGS.train_dir))
    read_flickrlogos27(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    main()
