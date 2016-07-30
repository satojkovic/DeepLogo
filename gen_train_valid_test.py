#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from scipy import ndimage
from six.moves import cPickle as pickle

CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32
CNN_IN_CH = 3
PIXEL_DEPTH = 255.0
TRAIN_DIR = 'flickr_logos_27_dataset'
CROPPED_AUG_IMAGE_DIR = os.path.join(
    TRAIN_DIR, 'flickr_logos_27_dataset_cropped_augmented_images')


def load_logo(data_dir):
    image_files = os.listdir(data_dir)
    dataset = np.ndarray(
        shape=(len(image_files), CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH),
        dtype=np.float32)
    print(data_dir)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(data_dir, image)
        try:
            image_data = (
                ndimage.imread(image_file).astype(float) - PIXEL_DEPTH / 2
            ) / PIXEL_DEPTH
            if image_data.shape != (CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH):
                raise Exception('Unexpected image shape: %s' %
                                str(image_data.shape))
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


def main():
    train_test_dirs = [
        os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, train_test_dir)
        for class_name in os.listdir(CROPPED_AUG_IMAGE_DIR)
        for train_test_dir in os.listdir(
            os.path.join(CROPPED_AUG_IMAGE_DIR, class_name))
    ]

    train_datasets = maybe_pickle(train_test_dirs[1::2])
    test_datasets = maybe_pickle(train_test_dirs[0::2])


if __name__ == '__main__':
    main()
