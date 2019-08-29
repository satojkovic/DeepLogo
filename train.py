#!/usr/bin/env python
# -*- coding=utf-8 -*-

import argparse
from darknet import Darknet
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train yolov3 based DeepLogo model')
    parser.add_argument('--cfg_file', type=str, default='cfg/yolov3.cfg', help='Path to yolov3.cfg file')
    parameters = parser.parse_args()
    print('[Parameters]')
    print(parameters)

    model = Darknet(parameters.cfg_file)

    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()