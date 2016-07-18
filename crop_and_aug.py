#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image
from collections import defaultdict

CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32

TRAIN_DIR = 'flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'flickr_logos_27_dataset_images')
CROPPED_IMAGE_DIR = os.path.join(TRAIN_DIR,
                                 'flickr_logos_27_dataset_cropped_images')



def crop_logos(annot_train):
    if not os.path.exists(CROPPED_IMAGE_DIR):
        os.makedirs(CROPPED_IMAGE_DIR)

    # Multiple images are cropped from same file.
    crop_per_files = defaultdict(int)
    for annot in annot_train:
        fn = annot[0].decode('utf-8')
        class_name = annot[1].decode('utf-8')
        train_subset_class = annot[2].decode('utf-8')

        crop_per_files[fn] += 1
        x1, y1, x2, y2 = list(map(int, annot[3:]))
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            print('Skipping:', fn)
            continue
        im = Image.open(os.path.join(TRAIN_IMAGE_DIR, fn))
        cropped_im = im.crop((x1, y1, x2, y2))
        resized_im = cropped_im.resize((CNN_IN_WIDTH, CNN_IN_HEIGHT))

        _, ext = os.path.splitext(fn)
        cropped_fn = '_'.join(
            [fn.split('.')[0], class_name, train_subset_class,
             str(crop_per_files[fn])]) + ext
        resized_im.save(os.path.join(CROPPED_IMAGE_DIR, cropped_fn))

    # check
    org_imgs = [img for img in os.listdir(TRAIN_IMAGE_DIR)]
    cropped_imgs = [img for img in os.listdir(CROPPED_IMAGE_DIR)]
    print('original: %d' % (len(org_imgs)))
    print('cropped: %d' % (len(cropped_imgs)))

    return cropped_imgs


def main():
    annot_train = np.loadtxt(
        os.path.join(TRAIN_DIR,
                     'flickr_logos_27_dataset_training_set_annotation.txt'),
        dtype='a')
    print('train_annotation: %d, %d ' % (annot_train.shape))

    #
    # crop logos from an image
    #
    cropped_imgs = crop_logos(annot_train)

if __name__ == '__main__':
    main()
