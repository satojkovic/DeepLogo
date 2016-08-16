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
from PIL import Image
import sys

CLASS_NAME = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex',
              'Ferrari', 'Ford', 'Google', 'HP', 'Heineken', 'Intel',
              'McDonalds', 'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma',
              'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone',
              'Yahoo']

FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_string(
    "test_dir", "flickr_logos_27_dataset/flickr_logos_27_dataset_images",
    "Directory")
tf.app.flags.DEFINE_integer("image_width", 64, "A width of an input image")
tf.app.flags.DEFINE_integer("image_height", 32, "A height of an input image")
tf.app.flags.DEFINE_integer("num_channels", 3,
                            "A number of channels of an input image")
tf.app.flags.DEFINE_integer("num_classes", 27, "Number of logo classes.")
tf.app.flags.DEFINE_integer("patch_size", 5,
                            "A patch size of convolution filter")


def model(data, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1,
          b_fc1, w_fc2, b_fc2):
    # First layer
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(
            data, w_conv1, [1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(
        h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second layer
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(
            h_pool1, w_conv2, [1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(
        h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    # Third layer
    h_conv3 = tf.nn.relu(
        tf.nn.conv2d(
            h_pool2, w_conv3, [1, 1, 1, 1], padding='SAME') + b_conv3)
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
        test_images_fn = [test_image
                          for test_image in os.listdir(FLAGS.test_dir)]
        test_image_fn = np.random.choice(test_images_fn, 1)[0]
        test_image_fn = os.path.join(FLAGS.test_dir, test_image_fn)
    print("Test image:", test_image_fn)

    # Open and resize a test image
    test_image = Image.open(test_image_fn)
    test_image = test_image.resize((FLAGS.image_width, FLAGS.image_height))
    test_image = np.reshape(test_image,
                            (1, FLAGS.image_width, FLAGS.image_height,
                             FLAGS.num_channels)).astype(np.float32)

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
            tf.truncated_normal(
                [16 * 4 * 128, 2048], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[2048]))

        w_fc2 = tf.Variable(tf.truncated_normal([2048, FLAGS.num_classes]))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_classes]))

        params = [w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1,
                  b_fc1, w_fc2, b_fc2]

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
        tf.initialize_all_variables().run()
        if initial_weights is not None:
            session.run(assign_ops)
            print('initialized by pre-learned values')
        else:
            print('initialized')
        pred = session.run([test_pred])
        print("Class name:", CLASS_NAME[np.argmax(pred)])


if __name__ == '__main__':
    main()
