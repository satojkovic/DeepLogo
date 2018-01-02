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

import os

__all__ = ('CLASS_NAME', 'CNN_IN_WIDTH', 'CNN_IN_HEIGHT', 'CNN_IN_CH',
           'CNN_SHAPE', 'TRAIN_DIR', 'TRAIN_IMAGE_DIR',
           'CROPPED_AUG_IMAGE_DIR', 'ANNOT_FILE', 'ANNOT_FILE_WITH_BG')

CLASS_NAME = [
    'Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari',
    'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc',
    'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks',
    'Texaco', 'Unicef', 'Vodafone', 'Yahoo', 'Background'
]

CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32
CNN_IN_CH = 3
CNN_SHAPE = (CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)

TRAIN_DIR = 'flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'flickr_logos_27_dataset_images')
CROPPED_AUG_IMAGE_DIR = os.path.join(
    TRAIN_DIR, 'flickr_logos_27_dataset_cropped_augmented_images')
ANNOT_FILE = os.path.join(
    TRAIN_DIR, 'flickr_logos_27_dataset_training_set_annotation.txt')
ANNOT_FILE_WITH_BG = os.path.join(TRAIN_DIR, 'train_annot_with_bg_class.txt')
