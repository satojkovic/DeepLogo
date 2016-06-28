#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image

TRAIN_DIR = 'flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'flickr_logos_27_dataset_images')
CROPPED_IMAGE_DIR = os.path.join(
    TRAIN_DIR, 'flickr_logos_27_dataset_cropped_32x32_images')

CNN_IMAGE_SIZE = 32


def main():
    annot_train = np.loadtxt(
        os.path.join(TRAIN_DIR,
                     'flickr_logos_27_dataset_training_set_annotation.txt'),
        dtype=np.str)
    print('train_annotation: %d, %d ' % (annot_train.shape))

    #
    # crop logo from an image
    #

    if not os.path.exists(CROPPED_IMAGE_DIR):
        os.makedirs(CROPPED_IMAGE_DIR)

    for annot in annot_train:
        x1, y1, x2, y2 = annot[3:]
        im = Image.open(os.path.join(TRAIN_DIR, annot[0]))

        cx, cy = (x2 - x1) // 2, (y2 - y1) // 2
        cropped_im = im.crop(cx - (CNN_IMAGE_SIZE // 2), cy -
                             (CNN_IMAGE_SIZE // 2), cx + (CNN_IMAGE_SIZE // 2),
                             cy + (CNN_IMAGE_SIZE // 2))

        _, ext = os.path.splitext(annot[0])
        cropped_fn = '_'.join([annot[0].split('.')[0], '32x32']) + ext
        im.save(os.path.join(CROPPED_IMAGE_DIR, cropped_fn))

    # check
    org_imgs = [img for img in os.listdir(TRAIN_DIR)]
    cropped_imgs = [img for img in os.listdir(CROPPED_IMAGE_DIR)]
    print('original: %d' % (len(org_imgs)))
    print('cropped: %d' % (len(cropped_imgs)))


if __name__ == '__main__':
    main()
