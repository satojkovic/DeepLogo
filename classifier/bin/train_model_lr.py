#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
import cv2
import numpy as np
import os
from sklearn.externals import joblib
import scipy.cluster.vq as vq


def make_vw_histogram(img_dir, train_label_file, codebook):
    train_imgfn_labels = np.loadtxt(train_label_file,
                                    delimiter='\t',
                                    dtype=str)

    vw_hists = []
    labels = []
    for imgfn, label in train_imgfn_labels:
        # extract SIFT feature
        img = cv2.imread(os.path.join(img_dir, imgfn))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray, None)
        if len(kp):
            code, dist = vq.vq(des, codebook)
            vw_hist, bin_edges = np.histogram(code,
                                              bins=codebook.shape[0],
                                              normed=True)
            vw_hists.append(vw_hist)
            labels.append(label)
    return vw_hists, labels


def main():
    CROPPED_IMAGE_DIR = "../../logo_images/cropped"
    TRAIN_LABEL_FILE = "../data/logo_train_labels_sorted.tsv"
    CODEBOOK_FILE = "cdbk.pkl"

    # load codebook
    codebook = joblib.load(CODEBOOK_FILE)

    # visual words histogram
    vw_hists, labels = make_vw_histogram(CROPPED_IMAGE_DIR,
                                         TRAIN_LABEL_FILE,
                                         codebook)

    # train logistic regression model
    logreg = LogisticRegression(C=1e5)
    logreg.fit(vw_hists, labels)
    joblib.dump(logreg, "logreg_learned.pkl")

if __name__ == '__main__':
    main()
