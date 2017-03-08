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
from PIL import Image
from collections import defaultdict
from itertools import product
from sklearn.model_selection import train_test_split
import shutil
import re
import glob

CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32

DATA_AUG_POS_SHIFT_MIN = -2
DATA_AUG_POS_SHIFT_MAX = 2
DATA_AUG_SCALES = [0.9, 1.1]
DATA_AUG_ROT_MIN = -15
DATA_AUG_ROT_MAX = 15

TRAIN_DIR = 'flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'flickr_logos_27_dataset_images')
CROPPED_AUG_IMAGE_DIR = os.path.join(
    TRAIN_DIR, 'flickr_logos_27_dataset_cropped_augmented_images')
ANNOT_FILE = 'flickr_logos_27_dataset_training_set_annotation.txt'
NONE_IMAGE_DIR = os.path.join(TRAIN_DIR, 'SUN397')
NUM_OF_NONE_IMAGES = 10000


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
    for sx, sy in product(
            range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX),
            range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX)):
        cx = rect['cx'] + sx
        cy = rect['cy'] + sy
        cropped_im = im.crop((cx - rect['wid'] // 2, cy - rect['hgt'] // 2,
                              cx + rect['wid'] // 2, cy + rect['hgt'] // 2))
        resized_im = cropped_im.resize((CNN_IN_WIDTH, CNN_IN_HEIGHT))
        aug_pos_ims.append(resized_im)
        aug_pos_suffixes.append('p' + str(sx) + str(sy))
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
        cropped_im = rotated_im.crop(
            (rect['cx'] - rect['wid'] // 2, rect['cy'] - rect['hgt'] // 2,
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
    return [cropped_im], [cropped_suffix]


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
    dst_dir = os.path.join(CROPPED_AUG_IMAGE_DIR, class_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i, arg in enumerate(args):
        for im, suffix in zip(arg[0], arg[1]):
            save_fn = '_'.join([
                fn.split('.')[0], class_name, train_subset_class, str(cnt),
                suffix
            ]) + os.path.splitext(fn)[1]
            im.save(os.path.join(dst_dir, save_fn))


def close_im(*args):
    for ims in args:
        for im in ims:
            im.close()


def crop_and_aug(annot_train):
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
        cropped_ims, cropped_suffixes = crop_logos(annot, im)

        # augment by shifting a center
        shifted_ims, shifted_suffixes = aug_pos(annot, im)

        # augment by scaling
        scaled_ims, scaled_suffixes = aug_scale(annot, im)

        # augment by rotation
        rotated_ims, rotated_suffixes = aug_rot(annot, im)

        # save images
        save_im(annot, cnt_per_file[fn], [cropped_ims, cropped_suffixes],
                [shifted_ims, shifted_suffixes], [scaled_ims, scaled_suffixes],
                [rotated_ims, rotated_suffixes])

        # close image file
        close_im([im], cropped_ims, shifted_ims, scaled_ims, rotated_ims)


def crop_none():
    none_img_classes = [
        cn.decode('utf-8')
        for cn in np.loadtxt(
            os.path.join(NONE_IMAGE_DIR, 'ClassName.txt'), dtype='a')
    ]

    dst_dir = os.path.join(CROPPED_AUG_IMAGE_DIR, 'None')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for none_class in none_img_classes:
        none_dir = os.path.join(NONE_IMAGE_DIR, none_class[1:])
        none_imgs = [
            os.path.join(none_dir, img) for img in os.listdir(none_dir)
            if re.search('\.jpg', img)
        ]
        none_imgs = np.random.choice(none_imgs, 10)
        for none_img in none_imgs:
            im = Image.open(none_img)
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            cw, ch = w // 2, h // 2
            cropped_im = im.crop(
                (cw - CNN_IN_WIDTH // 2, ch - CNN_IN_HEIGHT // 2,
                 cw + CNN_IN_WIDTH // 2, ch + CNN_IN_HEIGHT // 2))
            dst_fn = os.path.basename(none_img)
            cropped_im.save(os.path.join(dst_dir, dst_fn))


def crop_and_aug_with_none(annot_train, with_none=False):
    # root directory to save processed images
    if not os.path.exists(CROPPED_AUG_IMAGE_DIR):
        os.makedirs(CROPPED_AUG_IMAGE_DIR)

    # crop images and apply augmentation
    crop_and_aug(annot_train)

    # crop images of none class
    if with_none:
        crop_none()

    # print results
    org_imgs = [img for img in os.listdir(TRAIN_IMAGE_DIR)]
    crop_and_aug_imgs = [
        fname
        for root, dirs, files in os.walk(CROPPED_AUG_IMAGE_DIR)
        for fname in glob.glob(os.path.join(root, '*.jpg'))
    ]
    print('original: %d' % (len(org_imgs)))
    print('cropped: %d' % (len(crop_and_aug_imgs)))


def do_train_test_split():
    class_names = [cls for cls in os.listdir(CROPPED_AUG_IMAGE_DIR)]
    for class_name in class_names:
        if os.path.exists(
                os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, 'train')):
            continue
        if os.path.exists(
                os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, 'test')):
            continue

        imgs = [
            img
            for img in os.listdir(
                os.path.join(CROPPED_AUG_IMAGE_DIR, class_name))
        ]
        # train=0.75, test=0.25
        train_imgs, test_imgs = train_test_split(imgs)
        # move images to train or test directory
        os.makedirs(os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, 'train'))
        os.makedirs(os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, 'test'))
        for img in train_imgs:
            dst = os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, 'train')
            src = os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, img)
            shutil.move(src, dst)
        for img in test_imgs:
            dst = os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, 'test')
            src = os.path.join(CROPPED_AUG_IMAGE_DIR, class_name, img)
            shutil.move(src, dst)


def main():
    annot_train = np.loadtxt(os.path.join(TRAIN_DIR, ANNOT_FILE), dtype='a')
    print('train_annotation: %d, %d ' % (annot_train.shape))

    # cropping and data augmentation
    crop_and_aug_with_none(annot_train)

    # train_test_split
    do_train_test_split()


if __name__ == '__main__':
    main()
