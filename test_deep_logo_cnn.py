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
import cv2
import skimage.io
import skimage.transform
import common
import model
import preprocess
from scipy.misc import imresize

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
            os.path.join(common.CROPPED_AUG_IMAGE_DIR, class_name, 'test')
            for class_name in common.CLASS_NAME
        ]
        test_dir = np.random.choice(test_dirs)
        test_images_fn = [test_image for test_image in os.listdir(test_dir)]
        test_image_fn = np.random.choice(test_images_fn, 1)[0]
        test_image_fn = os.path.join(test_dir, test_image_fn)
    print("Test image:", test_image_fn)

    # Open and resize a test image
    if common.CNN_IN_CH == 1:
        test_image_org = skimage.io.imread(test_image_fn, as_grey=True)
        test_image_org = test_image_org.reshape(
            common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH, common.CNN_IN_CH)
    else:
        test_image_org = skimage.io.imread(test_image_fn)
    if test_image_org.shape != (common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
                                common.CNN_IN_CH):
        test_image_org = imresize(
            test_image_org, (common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH),
            interp='bicubic')
    test_image_org = preprocess.scaling(test_image_org)
    test_image = test_image_org.reshape(
        (1, common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
         common.CNN_IN_CH)).astype(np.float32)

    # Training model
    graph = tf.Graph()
    with graph.as_default():
        # Weights and biases
        model_params = model.params()

        # restore weights
        f = "weights.npz"
        if os.path.exists(f):
            initial_weights = load_initial_weights(f)
        else:
            initial_weights = None

        if initial_weights is not None:
            assert len(initial_weights) == len(model_params)
            assign_ops = [
                w.assign(v) for w, v in zip(model_params, initial_weights)
            ]

        # A placeholder for a test image
        tf_test_image = tf.constant(test_image)

        # model
        logits = model.cnn(tf_test_image, model_params, keep_prob=1.0)
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
