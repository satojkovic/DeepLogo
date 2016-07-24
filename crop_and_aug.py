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
DATA_AUG_SCALES = [0.9, 1.1]
DATA_AUG_ROT_MIN = -15
DATA_AUG_ROT_MAX = 15

TRAIN_DIR = 'flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'flickr_logos_27_dataset_images')
CROPPED_IMAGE_DIR = os.path.join(TRAIN_DIR,
                                 'flickr_logos_27_dataset_cropped_images')


def parse_annot(annot):
    fn = annot[0].decode('utf-8')
    class_name = annot[1].decode('utf-8')
    train_subset_class = annot[2].decode('utf-8')
    return fn, class_name, train_subset_class

def get_rect(annot):
    rect = defaultdict(int)
    x1, y1, x2, y2 = rect_coord(annot[3:])
    cx, cy, wid, hgt = center_wid_hgt(x1, y1, x2, y2)
    rect['x1'] = x1
    rect['y1'] = y1
    rect['x2'] = x2
    rect['y2'] = y2
    rect['cx'] = cx
    rect['cy'] = cy
    rect['wid'] = wid
    rect['hgt'] = hgt
    return rect


def aug_pos(annot, im):
    aug_pos_ims = []
    aug_pos_suffixes = []

    rect = get_rect(annot)
    for sx, sy in product(range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX), range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX)):
        cx = rect['cx'] + sx
        cy = rect['cy'] + sy
        cropped_im = im.crop((cx - rect['wid'] // 2, cy - rect['hgt'] // 2,
                              cx + rect['wid'] // 2, cy + rect['hgt'] // 2))
        resized_im = cropped_im.resize((CNN_IN_WIDTH, CNN_IN_HEIGHT))
        aug_pos_ims.append(resized_im)
        aug_pos_suffixes.append('p' + str(cx) + str(cy))
        cropped_im.close()

    return aug_pos_ims, aug_pos_suffixes


def aug_scale(annot, im):
    aug_scale_ims = []
    aug_scale_suffixes = []

    rect = get_rect(annot)
    for s in DATA_AUG_SCALES:
        w = int(rect['wid'] * s)
        h = int(rect['hgt'] * s)
        cropped_im = im.crop((rect['cx'] - w // 2, rect['cy'] - h // 2,
                              rect['cx'] + w // 2, rect['cy'] + h // 2))
        resized_im = cropped_im.resize((CNN_IN_WIDTH, CNN_IN_HEIGHT))
        aug_scale_ims.append(resized_im)
        aug_scale_suffixes.append('s' + str(s))
        cropped_im.close()

    return aug_scale_ims, aug_scale_suffixes


def aug_rot(annot, im):
    aug_rot_ims = []
    aug_rot_suffixes = []

    rect = get_rect(annot)
    for r in range(DATA_AUG_ROT_MIN, DATA_AUG_ROT_MAX):
        rotated_im = im.rotate(r)
        cropped_im = rotated_im.crop((rect['cx'] - rect['wid'] // 2, rect['cy'] - rect['hgt'] // 2,
                                      rect['cx'] + rect['wid'] // 2, rect['cy'] + rect['hgt'] // 2))
        resized_im = cropped_im.resize((CNN_IN_WIDTH, CNN_IN_HEIGHT))
        aug_rot_ims.append(resized_im)
        aug_rot_suffixes.append('r' + str(r))
        rotated_im.close()
        cropped_im.close()

    return aug_rot_ims, aug_rot_suffixes


def crop_logos(annot, im):
    x1, y1, x2, y2 = rect_coord(annot[3:])
    cropped_im = im.crop((x1, y1, x2, y2))
    cropped_im = cropped_im.resize((CNN_IN_WIDTH, CNN_IN_HEIGHT))
    cropped_suffix = 'p00'
    return cropped_im, cropped_suffix


def rect_coord(annot_part):
    return list(map(int, annot_part))


def center_wid_hgt(x1, y1, x2, y2):
    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2
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


def save_im(annot, cnt, *args):
    fn, class_name, train_subset_class = parse_annot(annot)
    for arg in args:
        im, suffix = arg
        save_fn = '_'.join([fn.split('.')[0], class_name,
                            train_subset_class, str(cnt), suffix]) + os.path.splitext(fn)[1]
        im.save(os.path.join(CROPPED_IMAGE_DIR, save_fn))

def close_im(*args):
    for im in args:
        im.close()


def crop_and_aug(annot_train):
    if not os.path.exists(CROPPED_IMAGE_DIR):
        os.makedirs(CROPPED_IMAGE_DIR)

    cnt_per_file = defaultdict(int)
    for annot in annot_train:
        # for generating a file name
        fn, _, _ = parse_annot(annot)
        cnt_per_file[fn] += 1
        
        # skip if width or height equal zero
        if is_skip(annot[3:]):
            print('Skip: ', fn)
            continue

        # open an image
        im = Image.open(os.path.join(TRAIN_IMAGE_DIR, fn))

        # normal cropping
        cropped_im, cropped_suffix = crop_logos(annot, im)

        # augment by shifting a center
        shifted_ims, shifted_suffixes = aug_pos(annot, im)

        # augment by scaling
        scaled_ims, scaled_suffixes = aug_scale(annot, im)
        
        # augment by rotation
        rotated_im, rotated_suffix = aug_rot(annot, im)

        # save images
        save_im(annot, cnt_per_file[fn],
                [cropped_im, cropped_suffix],
                [shifted_ims, shifted_suffixes],
                [scaled_ims, scaled_suffixes],
                [rotated_ims, rotated_suffixes])

        # close image file
        close_im(im, cropped_im, shifted_ims, scaled_ims, rotated_ims)

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
