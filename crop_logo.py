#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os

TRAIN_DIR = 'flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'flickr_logos_27_dataset_images')


def main():
    annot_train = np.loadtxt(
        os.path.join(TRAIN_DIR,
                     'flickr_logos_27_dataset_training_set_annotation.txt'),
        dtype=np.str)
    print('train_annotation: %d, %d ' % (annot_train.shape))


if __name__ == '__main__':
    main()
