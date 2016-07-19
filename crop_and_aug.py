#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image
from collections import defaultdict
from itertools import product

CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32

DATA_AUG_POS_SHIFT_MIN = -2
DATA_AUG_POS_SHIFT_MAX = 2

TRAIN_DIR = 'flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'flickr_logos_27_dataset_images')
CROPPED_IMAGE_DIR = os.path.join(TRAIN_DIR,
                                 'flickr_logos_27_dataset_cropped_images')


def parse_annot(annot):
    fn = annot[0].decode('utf-8')
    class_name = annot[1].decode('utf-8')
    train_subset_class = annot[2].decode('utf-8')
    return fn, class_name, train_subset_class


def aug_pos(annot_train, im):
    for annot in annot_train:
        fn, class_name, train_subset_class = parse_annot(annot)
        x1, y1, x2, y2 = list(map(int, annot[3:]))
        cx = (x2 - x1) // 2
        cy = (y2 - y1) // 2
        wid, hgt = (x2 - x1), (y2 - y1)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            print('Skipping:', fn)
            continue
        im = Image.open(os.path.join(TRAIN_IMAGE_DIR, fn))
        
        for sx, sy in product(range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX), range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX)):
            cx = cx + sx
            cy = cy + sy
            cropped_im = im.crop((cx - wid//2, cy - hgt//2, cx + wid//2, cy + hgt//2))
            resized_im = cropped_im.resize(CNN_IN_WIDTH, CNN_IN_HEIGHT)
            

def aug_scale(annot_train, im):
    pass


def aug_rot(annot_train, im):
    pass


def crop_logos(annot, im):
    x1, y1, x2, y2 = rect_coord(annot[3:])
    cropped_im = im.crop((x1, y1, x2, y2))
    cropped_im = cropped_im.resize((CNN_IN_WIDTH, CNN_IN_HEIGHT))
    return cropped_im


def rect_coord(annot_part):
    return list(map(int, annot_part))


def center_wid_hgt(x1, y1, x2, y2):
    cx = (x2 - x1) // 2
    cy = (y2 - y1) // 2
    wid = (x2 - x1)
    hgt = (y2 - y1)
    return cx, cy, wid, hgt


def is_skip(annot_part):
    x1, y1, x2, y2 = rect_coord(annot_part)
    _, _, wid, hgt = center_wid_hgt(x1, y1, x2, y2)
    if wid <= 0 or hgt <= 0:
        return True
    else:
        return False


def crop_and_aug(annot_train):
    if not os.path.exists(CROPPED_IMAGE_DIR):
        os.makedirs(CROPPED_IMAGE_DIR)

    for annot in annot_train:
        # for generating a file name
        fn, class_name, train_subset_class = parse_annot(annot)
        
        # skip if width or height equal zero
        if is_skip(annot[3:]):
            print('Skip: ', fn)
            continue

        # open an image
        im = Image.open(os.path.join(TRAIN_IMAGE_DIR, fn))

        # normal cropping
        cropped_im = crop_logos(annot, im)

        # augment by shifting a center
        aug_pos(annot, im)

        # augment by scaling
        aug_scale(annot, im)
        
        # augment by rotation
        aug_rot(annot, im)

        # close image file
        im.close()
        cropped_im.close()
        

    # print results
    org_imgs = [img for img in os.listdir(TRAIN_IMAGE_DIR)]
    crop_and_aug_imgs = [img for img in os.listdir(CROPPED_IMAGE_DIR)]
    print('original: %d' % (len(org_imgs)))
    print('cropped: %d' % (len(crop_and_aug_imgs)))

    return crop_and_aug_imgs


def main():
    annot_train = np.loadtxt(
        os.path.join(TRAIN_DIR,
                     'flickr_logos_27_dataset_training_set_annotation.txt'),
        dtype='a')
    print('train_annotation: %d, %d ' % (annot_train.shape))

    # normal cropping and data augmentation
    crop_and_aug(annot_train)

if __name__ == '__main__':
    main()
