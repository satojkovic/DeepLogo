#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import numpy as np
import scipy.cluster.vq as vq
import cv2
import os
from sklearn.metrics import accuracy_score, confusion_matrix


def make_vw_histogram(img_dir, query_file, codebook):
    query_imgfn_labels = np.loadtxt(query_file,
                                    delimiter='\t',
                                    dtype=str)

    vw_hists = []
    labels = []
    for imgfn, label in query_imgfn_labels:
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


def classify(logreg, test_vw_hists, test_labels):
    print '# ---- classification report ---- #'
    predlabels = logreg.predict(test_vw_hists)

    print '## accuracy: %s' % (accuracy_score(test_labels, predlabels))
    cm = confusion_matrix(test_labels, predlabels)
    print '## confusion matrix'
    print cm


def main():
    IMAGE_DIR = "../../logo_images/cropped"
    QUERY_FILE = "../data/logo_train_labels_sorted.tsv"
    LOGREG_MODEL_FILE = "logreg_learned.pkl"
    CODEBOOK_FILE = "cdbk.pkl"

    # load codebook
    codebook = joblib.load(CODEBOOK_FILE)

    # load logistic regression model
    logreg = joblib.load(LOGREG_MODEL_FILE)

    # visual words histogram
    test_vw_hists, test_labels = make_vw_histogram(IMAGE_DIR,
                                                   QUERY_FILE,
                                                   codebook)

    # classification report
    classify(logreg, test_vw_hists, test_labels)

if __name__ == '__main__':
    main()
