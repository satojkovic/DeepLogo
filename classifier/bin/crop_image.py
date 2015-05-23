#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from collections import defaultdict


def crop_logo_and_assign_label(img_dir, cropped_img_dir, annot_file, labeldata, labelfilename):
    labelfile = open(labelfilename, 'w')

    # ensure cropped image directory
    if not os.path.isdir(cropped_img_dir):
        os.makedirs(cropped_img_dir)
    
    # <image filename> <brand> <x1> <y1> <x2> <y2>
    annot_data = np.genfromtxt(annot_file, delimiter=' ', dtype=str)
    print 'Num of annotated images:', len(annot_data)

    # crop logos from images
    brand_img_ids = defaultdict(int)
    for data in annot_data:
        fname = data[0]
        brand = data[1]
        brand_img_ids[brand] += 1
        x1, y1, x2, y2 = int(data[2]), int(data[3]), int(data[4]), int(data[5])

        img = cv2.imread(os.path.join(img_dir, fname))
        cropped_img = img[y1:y2, x1:x2]

        # save a cropped image
        name, ext = os.path.splitext(fname)
        fname = '_'.join([name, str(brand_img_ids[brand])]) + ext
        cv2.imwrite(os.path.join(cropped_img_dir, fname), cropped_img)

        # save a labelfile entry
        labelstr = '%s\t%s\n' % (fname, labeldata[brand])
        labelfile.write(labelstr)

    labelfile.close()


def readlabels(filename):
    import csv
    labeldata = {}
    reader = csv.reader(open(filename, 'r'), delimiter='\t')
    for line in reader:
        labeldata[line[0]] = int(line[1])
    return labeldata

if __name__ == '__main__':
    IMAGE_DIR = "../../logo_images"
    CROPPED_IMAGE_DIR = "../../logo_images/cropped"
    ANNOT_FILE = "../data/logo_annot.txt"
    TRAIN_LABEL_FILE = "../data/logo_train_labels.tsv"
    labeldata = readlabels("../data/logo_label.tsv")

    crop_logo_and_assign_label(IMAGE_DIR, CROPPED_IMAGE_DIR, ANNOT_FILE,
                               labeldata, TRAIN_LABEL_FILE)
