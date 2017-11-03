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

import tensorflow as tf
import numpy as np
import os
import sys
from scipy import ndimage
import re
import common

TRAIN_DIR = 'flickr_logos_27_dataset'
CROPPED_AUG_IMAGE_DIR = os.path.join(
    TRAIN_DIR, 'flickr_logos_27_dataset_cropped_augmented_images')
PIXEL_DEPTH = 255.0

FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_integer("image_width", common.CNN_IN_WIDTH,
                            "A width of an input image")
tf.app.flags.DEFINE_integer("image_height", common.CNN_IN_HEIGHT,
                            "A height of an input image")
tf.app.flags.DEFINE_integer("num_channels", common.CNN_IN_CH,
                            "A number of channels of an input image")
tf.app.flags.DEFINE_integer("num_classes", 27, "Number of logo classes.")
tf.app.flags.DEFINE_integer("patch_size", 5,
                            "A patch size of convolution filter")


def model(data, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1,
          b_fc1, w_fc2, b_fc2):
    # First layer
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(data, w_conv1, [1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(
        h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second layer
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(h_pool1, w_conv2, [1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(
        h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    # Third layer
    h_conv3 = tf.nn.relu(
        tf.nn.conv2d(h_pool2, w_conv3, [1, 1, 1, 1], padding='SAME') + b_conv3)
    h_pool3 = tf.nn.max_pool(
        h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    conv_layer_flat = tf.reshape(h_pool3, [-1, 16 * 4 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, w_fc1) + b_fc1)

    # Output layer
    out = tf.matmul(h_fc1, w_fc2) + b_fc2

    return out


def load_initial_weights(fn):
    f = np.load(fn)

    # f.files has unordered keys ['arr_8', 'arr_9', 'arr_6'...]
    # Sorting keys by value of numbers
    initial_weights = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    return initial_weights


def main():
    if len(sys.argv) > 1:
        test_image_fn = sys.argv[1]
        if not os.path.exists(test_image_fn):
            print("Not found:", test_image_fn)
            sys.exit(-1)
    else:
        # Select a test image from a test directory
        test_dirs = [
            os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, 'test')
            for class_name in common.CLASS_NAME
        ]
        test_dir = np.random.choice(test_dirs)
        test_images_fn = [test_image for test_image in os.listdir(test_dir)]
        test_image_fn = np.random.choice(test_images_fn, 1)[0]
        test_image_fn = os.path.join(test_dir, test_image_fn)
    print("Test image:", test_image_fn)

    # Open and resize a test image
    test_image_org = (ndimage.imread(test_image_fn).astype(np.float32) -
                      PIXEL_DEPTH / 2) / PIXEL_DEPTH
    test_image_org.resize((common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
                           common.CNN_IN_CH))
    test_image = test_image_org.reshape(
        (1, common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH, common.CNN_IN_CH))

    # Training model
    graph = tf.Graph()
    with graph.as_default():
        # Variables
        w_conv1 = tf.Variable(
            tf.truncated_normal(
                [FLAGS.patch_size, FLAGS.patch_size, FLAGS.num_channels, 48],
                stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[48]))

        w_conv2 = tf.Variable(
            tf.truncated_normal(
                [FLAGS.patch_size, FLAGS.patch_size, 48, 64], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

        w_conv3 = tf.Variable(
            tf.truncated_normal(
                [FLAGS.patch_size, FLAGS.patch_size, 64, 128], stddev=0.1))
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))

        w_fc1 = tf.Variable(
            tf.truncated_normal([16 * 4 * 128, 2048], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[2048]))

        w_fc2 = tf.Variable(tf.truncated_normal([2048, FLAGS.num_classes]))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_classes]))

        params = [
            w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1,
            w_fc2, b_fc2
        ]

        # restore weights
        f = "weights.npz"
        if os.path.exists(f):
            initial_weights = load_initial_weights(f)
        else:
            initial_weights = None

        if initial_weights is not None:
            assert len(initial_weights) == len(params)
            assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

        # A placeholder for a test image
        tf_test_image = tf.constant(test_image)

        # model
        logits = model(tf_test_image, w_conv1, b_conv1, w_conv2, b_conv2,
                       w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2)
        test_pred = tf.nn.softmax(logits)

        # Restore ops
        saver = tf.train.Saver()

    # Recognize a brand logo of test image
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        if initial_weights is not None:
            session.run(assign_ops)
            print('initialized by pre-learned weights')
        elif os.path.exists("models"):
            save_path = "models/deep_logo_model"
            saver.restore(session, save_path)
            print('Model restored')
        else:
            print('initialized')
        pred = session.run([test_pred])
        print("Class name:", common.CLASS_NAME[np.argmax(pred)])
        print("Probability:", np.max(pred))


if __name__ == '__main__':
    main()
