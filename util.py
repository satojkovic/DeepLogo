# The MIT License (MIT)
# Copyright (c) 2017 satojkovic

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
import selectivesearch
import common
import skimage.io


def parse_annot(annot):
    fn = annot[0].decode('utf-8')
    class_name = annot[1].decode('utf-8')
    train_subset_class = annot[2].decode('utf-8')
    return fn, class_name, train_subset_class


def get_annot_rect(annot):
    return np.array(list(map(lambda x: int(x), annot[3:])))


def get_object_proposals(img, scale=500, sigma=0.9, min_size=10):
    # Selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=scale, sigma=sigma, min_size=min_size)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 500 pixels
        x, y, w, h = r['rect']
        if r['size'] < 2000 or w > 0.95 * img.shape[1] or h > 0.95 * img.shape[0]:
            continue
        # excluding the zero-width or zero-height box
        if r['rect'][2] == 0 or r['rect'][3] == 0:
            continue
        # distorted rects
        if w / h > 5 or h / w > 5:
            continue
        candidates.add(r['rect'])

    return candidates


def load_target_image(img_fn):
    if common.CNN_IN_CH == 1:
        target_image = skimage.io.imread(img_fn, as_grey=True)
    else:
        target_image = skimage.io.imread(img_fn)
    return target_image


def update_idx(results):
    probs = np.array([r['pred_prob'] for r in results])
    idx = np.argsort(probs)[::-1]
    return idx


def nms(recog_results, pred_prob_th=0.99, iou_th=0.5):
    # nms results
    nms_results = []

    # Discard all results with prob <= pred_prob_th
    pred_probs = np.array([r['pred_prob'] for r in recog_results])
    cand_idx = np.where(pred_probs > pred_prob_th)[0]
    cand_results = np.array(recog_results)[cand_idx]
    if len(cand_results) == 0:
        return nms_results

    # Sort in descending order
    cand_nms_idx = update_idx(cand_results)

    #
    # [Non-max suppression]
    #

    # Pick the result with the largest prob as a prediction
    pred = cand_results[cand_nms_idx[0]]
    nms_results.append(pred)
    if len(cand_results) == 1:
        return nms_results
    cand_results = cand_results[cand_nms_idx[1:]]
    cand_nms_idx = update_idx(cand_results)

    # Discard any remaining results with IoU >= iou_th
    while len(cand_results) > 0:
        del_idx = []
        del_seq_idx = []
        for seq_i, i in enumerate(cand_nms_idx):
            if iou_xywh(cand_results[i]['obj_proposal'],
                        pred['obj_proposal']) >= iou_th:
                del_idx.append(i)
                del_seq_idx.append(seq_i)
        # Delete non-max results
        cand_results = np.delete(cand_results, del_idx)
        if len(cand_results) == 0:
            break
        cand_nms_idx = update_idx(cand_results)
        # For next iteration
        pred, cand_results = cand_results[cand_nms_idx[0]], cand_results[
            cand_nms_idx[1:]]
        if len(cand_results) == 0:
            break
        cand_nms_idx = update_idx(cand_results)
        nms_results.append(pred)

    return nms_results


def iou(obj_proposal, annot_rect):
    """

    Arguments:
    obj_proposals -- rectangles of object proposals with coordinates (x, y, w, h)
    annot_rect -- rectangle of ground truth with coordinates (x1, y1, x2, y2)
    """
    xi1 = max(obj_proposal[0], annot_rect[0])
    yi1 = max(obj_proposal[1], annot_rect[1])
    xi2 = min(obj_proposal[0] + obj_proposal[2], annot_rect[2])
    yi2 = min(obj_proposal[1] + obj_proposal[3], annot_rect[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)

    # Calculate the union area by using formula: union(A, B) = A + B - inter_area
    box1_area = obj_proposal[2] * obj_proposal[3]
    box2_area = (annot_rect[2] - annot_rect[0]) * (
        annot_rect[3] - annot_rect[1])
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou


def iou_xywh(box1, box2):
    """

    Arguments:
    box1 -- rectangles of object proposals with coordinates (x, y, w, h)
    box2 -- rectangle of ground truth with coordinates (x1, y1, w, h)
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[0] + box1[2], box2[0] + box2[2])
    yi2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)

    # Calculate the union area by using formula: union(A, B) = A + B - inter_area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou
