#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import scipy.cluster.vq as vq
from collections import defaultdict


EXTENSIONS = [".jpg", ".bmp", ".png"]


def get_features_labels(img_dir, train_imgfn_labels):
    features = []
    labels = []
    imgfn2feat = defaultdict(list)
    for imgfn, label in train_imgfn_labels:
        # extract SIFT feature
        img = cv2.imread(os.path.join(img_dir, imgfn))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray, None)
        if len(kp):
            features.append(des)
            labels.append(label)
            imgfn2feat[imgfn].append(des)
    return features, labels, imgfn2feat


def compute_visual_words(features):
    X = np.vstack(features)
    num_of_feats = len(features)
    num_of_clusters = int(np.sqrt(num_of_feats))
    codebook, distortion = vq.kmeans(X,
                                     num_of_clusters,
                                     thresh=1e-05)
    return X, codebook, distortion


def compute_vw_histgoram(imgfn2feat, codebook):
    all_vw_histograms = defaultdict(list)
    for imgfn in sorted(imgfn2feat.keys()):
        X = np.vstack(imgfn2feat[imgfn])
        code, dist = vq.vq(X, codebook)
        vw_histogram, bin_edges = np.histogram(code,
                                               bins=range(codebook.shape[0]+1),
                                               normed=True)
        all_vw_histograms[imgfn] = vw_histogram
    return all_vw_histograms


def print_vw_histogram(all_vw_histograms):
    for imgfn, vw_histogram in all_vw_histograms.items():
        print imgfn, vw_histogram


def main():
    CROPPED_IMAGE_DIR = "../../logo_images/cropped"
    TRAIN_LABEL_FILE = "../data/logo_train_labels_sorted.tsv"

    train_imgfn_labels = np.loadtxt(TRAIN_LABEL_FILE,
                                    delimiter='\t',
                                    dtype=str)

    features, labels, imgfn2feat = get_features_labels(CROPPED_IMAGE_DIR,
                                                       train_imgfn_labels)

    # computing the visual words
    X, codebook, distortion = compute_visual_words(features)

    # computing the histogram of visual words
    all_vw_histograms = compute_vw_histgoram(imgfn2feat, codebook)
    print_vw_histogram(all_vw_histograms)

if __name__ == '__main__':
    main()
