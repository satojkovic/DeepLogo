#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "train_dir", "logo_images/cropped",
    "Directory where to write event logs and checkpoint.")


def train():
    pass


def main():
    if not tf.gfile.Exists(FLAGS.train_dir):
        print("Not found: %s" % (FLAGS.train_dir))
    train()


if __name__ == '__main__':
    main()
