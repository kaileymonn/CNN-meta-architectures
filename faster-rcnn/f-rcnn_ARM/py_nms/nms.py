#!/usr/bin/env python3
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = np.array(dets)[:, 0]
    y1 = np.array(dets)[:, 1]
    x2 = np.array(dets)[:, 2]
    y2 = np.array(dets)[:, 3]
    scores = np.array(dets)[:, 4]
    print(x1,"\n",y1,"\n",x2,"\n",y2,"\n",scores)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        print("intersection area = ", inter)
        print("union area = ", areas[i] + areas[order[1:]] - inter )
        print("iou is: ", ovr)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

"""Compare output with my C-implementation"""
NMS_THRESHOLD = 0.3
proposals = np.array([
    (12, 84, 140, 212, 0.5),
	(24, 84, 152, 212, 0.7),
	(36, 84, 164, 212, 0.88),
	(12, 96, 140, 224, 0.3),
	(24, 96, 152, 224, 0.66),
	(24, 108, 152, 236, 0.9)
    ])   
nms_output = py_cpu_nms(proposals, NMS_THRESHOLD)
print("Thresh = {0}, keep_index = {1}".format(NMS_THRESHOLD, nms_output))
    