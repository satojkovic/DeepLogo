#!/usr/bin/env python
# -*- coding=utf-8 -*-

import pandas as pd
from skimage import io
import config
import os
import warnings
import numpy as np

def main():
    # Load annot file
    logos_frame = pd.read_csv(config.ANNOT_FILE, header=None, delim_whitespace=True)
    print('Num. of annots:', len(logos_frame))

    if not os.path.exists(config.CROPPED_IMAGES_DIR):
        os.makedirs(config.CROPPED_IMAGES_DIR)

    num_cropped_images = 0
    annots = []
    for i, row in logos_frame.iterrows():
        img_name = row[0]
        cls_name = row[1]
        cls_idx = config.CLASS_NAMES.index(cls_name)
        subset = row[2]
        x1, y1, x2, y2 = row[3:]
        w, h = (x2 - x1), (y2 - y1)
        if w == 0 or h == 0:
            print('Skip:', img_name)
            continue
        img = io.imread(os.path.join(config.IMAGES_DIR, img_name))
        img_height, img_width, _ = img.shape
        x = (x1 + x2) / 2
        y = (y1 + y1) / 2
        annot = ','.join([img_name, str(x1), str(y1), str(x2), str(y2), str(cls_idx)])
        annots.append(annot)
        num_cropped_images += 1

    np.random.shuffle(annots)
    num_train = int(num_cropped_images * 0.8)
    with open(config.CROPPED_ANNOT_FILE, 'w') as f:
        for annot in annots[:num_train]:
            f.writelines(annot)
            f.writelines("\n")

    seen = set()
    num_test = 0
    with open(config.CROPPED_ANNOT_FILE_TEST, 'w') as f:
        for annot in annots[num_train:]:
            img_fn = annot.split(',')[0]
            if img_fn in seen:
                continue
            f.writelines(annot)
            f.writelines("\n")
            seen.add(img_fn)
            num_test += 1
    print('Num. of annotations: {}(train) {}(test)'.format(num_train, num_test))
    print('Created: {}'.format(config.CROPPED_ANNOT_FILE))
    print('Created: {}'.format(config.CROPPED_ANNOT_FILE_TEST))

if __name__ == "__main__":
    with warnings.catch_warnings():
        # Supress low contrast warnings
        warnings.simplefilter("ignore")

        # Crop logo images
        main()