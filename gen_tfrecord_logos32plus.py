#!/usr/bin/env python
# -*- coding=utf-8 -*-

import argparse
import os
import numpy as np
import io
from PIL import Image
import config
import scipy.io as sio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tensorflow as tf
from object_detection.utils import dataset_util


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_mat', required=True, help='Path to grountruth.mat')
    parser.add_argument('--img_dir', required=True, help='Path to image directory')
    return parser.parse_args()


def create_tf_example(img_fname, logo_name, bbox, img_dir, logo_names):
    x1, y1, w, h = list(map(int, bbox))
    x2, y2 = x1 + w, y1 + h
    cls_idx = logo_names[logo_name]
    cls_text = logo_name.encode('utf8')
    with tf.gfile.GFile(os.path.join(img_dir, img_fname), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    xmin = [x1 / width]
    xmax = [x2 / width]
    ymin = [y1 / height]
    ymax = [y2 / height]
    cls_text = [cls_text]
    cls_idx = [cls_idx]

    filename = img_fname.encode('utf8')
    image_format = b'jpg'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(cls_text),
        'image/object/class/label': dataset_util.int64_list_feature(cls_idx),        
    }))

    return tf_example


if __name__ == "__main__":
    args = parse_arguments()

    gts = sio.loadmat(args.gt_mat)
    logo_names = {gt[2][0] for gt in gts['groundtruth'][0]}
    logo_names = sorted(list(logo_names))
    logo_names = {name: i for i, name in enumerate(logo_names)}
    gt_train, gt_test = train_test_split(gts['groundtruth'][0])

    train_writer = tf.python_io.TFRecordWriter('train_logos32plus.tfrecord')
    for gt in tqdm(gt_train):
        img_fname = gt[0][0].replace('\\', '/')
        logo_name = gt[2][0]
        for bbox in gt[1]:
            tf_example = create_tf_example(img_fname, logo_name, bbox, args.img_dir, logo_names)
            train_writer.write(tf_example.SerializeToString())
    train_writer.close()

    test_writer = tf.python_io.TFRecordWriter('test_logos32plus.tfrecord')
    num_data = 0
    for gt in tqdm(gt_test):
        img_fname = gt[0][0].replace('\\', '/')
        logo_name = gt[2][0]
        for bbox in gt[1]:
            tf_example = create_tf_example(img_fname, logo_name, bbox, args.img_dir, logo_names)
            test_writer.write(tf_example.SerializeToString())
            break
        num_data += 1
    test_writer.close()
    print('Test ({} imgs)'.format(num_data))
