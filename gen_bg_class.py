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
import selectivesearch
import common
import util
import skimage.io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def get_bg_proposals(object_proposals, annot):
    annot_rect = util.get_annot_rect(annot)

    bg_proposals = []
    for obj_proposal in object_proposals:
        if iou(obj_proposal, annot_rect) <= 0.5:
            bg_proposals.append(obj_proposal)
    return bg_proposals


def get_object_proposals(img, scale=500, sigma=0.9, min_size=10):
    # Selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=scale, sigma=sigma, min_size=min_size)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # excluding large rectangle
        if r['rect'][0] + r['rect'][2] + 0.1 * img.shape[1] > img.shape[1] \
           or r['rect'][1] + r['rect'][3] + 0.1 * img.shape[0] > img.shape[0]:
            continue
        candidates.add(r['rect'])

    return candidates


def gen_annot_file_line(img_fn, class_name, train_subset_class, rect):
    rect = ' '.join(map(str, rect))
    line = ' '.join([img_fn, class_name, train_subset_class, rect])
    return line


def gen_annot_file_lines(annot):
    lines = []

    # Get original annot line
    img_fn, class_name, train_subset_class = util.parse_annot(annot)
    annot_rect = util.get_annot_rect(annot)
    lines.append(
        gen_annot_file_line(img_fn, class_name, train_subset_class,
                            annot_rect))

    # Load image
    img = skimage.io.imread(os.path.join(common.TRAIN_IMAGE_DIR, img_fn))

    # Selective search
    object_proposals = get_object_proposals(img)
    if len(object_proposals) == 0:
        return lines

    # Background proposals
    bg_proposals = get_bg_proposals(object_proposals, annot)
    if len(bg_proposals) == 0:
        return lines

    # Select bg proposal
    bg_proposal = bg_proposals[np.random.choice(
        np.array(bg_proposals).shape[0])]
    x1, y1, x2, y2 = bg_proposal[0], bg_proposal[
        1], bg_proposal[0] + bg_proposal[2], bg_proposal[1] + bg_proposal[3]
    lines.append(
        gen_annot_file_line(img_fn, common.CLASS_NAME[-1], train_subset_class,
                            [x1, y1, x2, y2]))

    return lines


def main():
    # Load an annotation file
    annot_train = np.loadtxt(common.ANNOT_FILE, dtype='a')
    print('train_annotation: {}'.format(annot_train.shape[0]))

    # Multi processing
    results = []
    n_workers = os.cpu_count()
    n_tasks = annot_train.shape[0]

    with ProcessPoolExecutor(n_workers) as executer, open(
            common.ANNOT_FILE_WITH_BG, 'w') as fw:
        for annot in annot_train:
            results.append(executer.submit(gen_annot_file_lines, annot))

        for result in as_completed(results):
            print('\n'.join(result.result()))
            fw.writelines('\n'.join(result.result()))
            fw.writelines('\n')


if __name__ == '__main__':
    main()
