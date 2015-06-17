#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import scipy.cluster.vq as vq
from sklearn.externals import joblib


EXTENSIONS = [".jpg", ".bmp", ".png"]


def get_features_labels(img_dir, train_label_file):
    train_imgfn_labels = np.loadtxt(train_label_file,
                                    delimiter='\t',
                                    dtype=str)
    features = []
    labels = []
    for imgfn, label in train_imgfn_labels:
        # extract SIFT feature
        img = cv2.imread(os.path.join(img_dir, imgfn))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray, None)
        if len(kp):
            features.append(des)
            labels.append(label)
    return features, labels


def compute_visual_words(features):
    X = np.vstack(features)
    num_of_feats = len(features)
    K = int(np.sqrt(num_of_feats))
    codebook, distortion = vq.kmeans(X, K, thresh=1e-05)
    return codebook, distortion


def main():
    CROPPED_IMAGE_DIR = "../../logo_images/cropped"
    TRAIN_LABEL_FILE = "../data/logo_train_labels_sorted.tsv"

    # get SIFT features
    features, labels = get_features_labels(CROPPED_IMAGE_DIR,
                                           TRAIN_LABEL_FILE)

    # compute the visual words
    codebook, distortion = compute_visual_words(features)

    # serialize codebook
    joblib.dump(codebook, 'cdbk.pkl')

if __name__ == '__main__':
    main()
