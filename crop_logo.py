#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image

CNN_IMAGE_SIZE = 32
postfix = str(CNN_IMAGE_SIZE) + 'x' + str(CNN_IMAGE_SIZE)

TRAIN_DIR = 'flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'flickr_logos_27_dataset_images')
CROPPED_IMAGE_DIR = os.path.join(
    TRAIN_DIR,
    '_'.join(['flickr_logos_27_dataset_cropped', postfix, 'images']))


def main():
    annot_train = np.loadtxt(
        os.path.join(TRAIN_DIR,
                     'flickr_logos_27_dataset_training_set_annotation.txt'),
        dtype='a')
    print('train_annotation: %d, %d ' % (annot_train.shape))

    #
    # crop logo from an image
    #

    if not os.path.exists(CROPPED_IMAGE_DIR):
        os.makedirs(CROPPED_IMAGE_DIR)

    for annot in annot_train:
        fn = annot[0].decode('utf-8')
        x1, y1, x2, y2 = list(map(int, annot[3:]))
        im = Image.open(os.path.join(TRAIN_IMAGE_DIR, fn))

        cx, cy = (x1 + x1) // 2, (y1 + y2) // 2
        cropped_im = im.crop(
            (cx - (CNN_IMAGE_SIZE // 2), cy - (CNN_IMAGE_SIZE // 2),
             cx + (CNN_IMAGE_SIZE // 2), cy + (CNN_IMAGE_SIZE // 2)))

        _, ext = os.path.splitext(fn)
        cropped_fn = '_'.join([fn.split('.')[0], postfix]) + ext
        cropped_im.save(os.path.join(CROPPED_IMAGE_DIR, cropped_fn))

    # check
    org_imgs = [img for img in os.listdir(TRAIN_IMAGE_DIR)]
    cropped_imgs = [img for img in os.listdir(CROPPED_IMAGE_DIR)]
    print('original: %d' % (len(org_imgs)))
    print('cropped: %d' % (len(cropped_imgs)))


if __name__ == '__main__':
    main()
