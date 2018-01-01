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
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # excluding large rectangle
        if r['rect'][0] + r['rect'][2] + 0.1 * img.shape[1] > img.shape[1] \
           or r['rect'][1] + r['rect'][3] + 0.1 * img.shape[0] > img.shape[0]:
            continue
        candidates.add(r['rect'])

    return candidates


def load_target_image(img_fn):
    if common.CNN_IN_CH == 1:
        target_image = skimage.io.imread(img_fn, as_grey=True)
    else:
        target_image = skimage.io.imread(img_fn)
    return target_image
