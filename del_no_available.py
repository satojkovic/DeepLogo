#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import shutil
import imagehash
from PIL import Image

TRAIN_DIR = 'flickr_logos_27_dataset'
DISTRACT_IMAGE_DIR = os.path.join(TRAIN_DIR,
                                  'flickr_logos_27_dataset_distractor_images')
NO_AVAILABLE_IMG = 'no_available.jpg'


def hash_value(img_fn, htype):
    img = Image.open(img_fn)
    if htype == 'a':
        hval = imagehash.average_hash(img)
    elif htype == 'p':
        hval = imagehash.phash(img)
    elif htype == 'd':
        hval = imagehash.dhash(img)
    elif htype == 'w':
        hval = imagehash.whash(img)
    else:
        hval = imagehash.average_hash(img)
    return hval


def main():
    imgs = [img for img in os.listdir(DISTRACT_IMAGE_DIR)
            if re.match('^(\d+)_', img)]
    src_hash = hash_value(
        os.path.join(DISTRACT_IMAGE_DIR, NO_AVAILABLE_IMG), htype='p')

    for img in imgs:
        target_hash = hash_value(
            os.path.join(DISTRACT_IMAGE_DIR, img), htype='p')
        if src_hash == target_hash:
            os.remove(os.path.join(DISTRACT_IMAGE_DIR, img))
            print('Delete:', img)


if __name__ == '__main__':
    main()
