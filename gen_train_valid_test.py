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

import numpy as np
import os
import skimage.io
from six.moves import cPickle as pickle
import common
import preprocess

PIXEL_DEPTH = 255.0
PICKLE_FILENAME = 'deep_logo.pickle'

TRAIN_SIZE = 70000
VALID_SIZE = 5000
TEST_SIZE = 5000


def load_logo(data_dir):
    image_files = os.listdir(data_dir)
    dataset = np.ndarray(
        shape=(len(image_files), common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
               common.CNN_IN_CH),
        dtype=np.float32)
    print(data_dir)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(data_dir, image)
        try:
            if common.CNN_IN_CH == 1:
                image_data = skimage.io.imread(image_file, as_grey=True)
                image_data = image_data.reshape(common.CNN_IN_HEIGHT,
                                                common.CNN_IN_WIDTH,
                                                common.CNN_IN_CH)
            else:
                image_data = skimage.io.imread(image_file)
            image_data = preprocess.scaling(image_data)
            if image_data.shape != (common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
                                    common.CNN_IN_CH):
                raise Exception(
                    'Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e,
                  '-it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_dirs, force=False):
    dataset_names = []
    for dir in data_dirs:
        set_filename = dir + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may overwrite by setting force=True
            print('%s already present - Skipping pickling. ' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_logo(dir)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names


def make_arrays(nb_rows, image_width, image_height, image_ch):
    if nb_rows:
        dataset = np.ndarray(
            (nb_rows, image_height, image_width, image_ch), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, common.CNN_IN_WIDTH,
                                              common.CNN_IN_HEIGHT,
                                              common.CNN_IN_CH)
    train_dataset, train_labels = make_arrays(train_size, common.CNN_IN_WIDTH,
                                              common.CNN_IN_HEIGHT,
                                              common.CNN_IN_CH)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                logo_set = pickle.load(f)
                np.random.shuffle(logo_set)
                if valid_dataset is not None:
                    valid_logo = logo_set[:vsize_per_class, :, :, :]
                    valid_dataset[start_v:end_v, :, :, :] = valid_logo
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                train_logo = logo_set[vsize_per_class:end_l, :, :, :]
                train_dataset[start_t:end_t, :, :, :] = train_logo
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels


def save_pickle(train_dataset, train_labels, valid_dataset, valid_labels,
                test_dataset, test_labels):
    try:
        f = open(PICKLE_FILENAME, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', PICKLE_FILENAME, ':', e)
        raise


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def main():
    train_dirs = [
        os.path.join(common.CROPPED_AUG_IMAGE_DIR, class_name, 'train')
        for class_name in common.CLASS_NAME
    ]
    test_dirs = [
        os.path.join(common.CROPPED_AUG_IMAGE_DIR, class_name, 'test')
        for class_name in common.CLASS_NAME
    ]

    train_datasets = maybe_pickle(train_dirs)
    test_datasets = maybe_pickle(test_dirs)

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, TRAIN_SIZE, VALID_SIZE)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, TEST_SIZE)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)

    save_pickle(train_dataset, train_labels, valid_dataset, valid_labels,
                test_dataset, test_labels)
    statinfo = os.stat(PICKLE_FILENAME)
    print('Compressed pickle size:', statinfo.st_size)


if __name__ == '__main__':
    main()
