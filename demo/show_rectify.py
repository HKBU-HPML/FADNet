#!/usr/bin/env python
import cv2
import numpy as np
import argparse
import time

DATAPATH='/media/sf_Shared_Data/gpuhomedataset/dispnet/custom'

def add_lines(img):
    height, width = img.shape[:2]
    interval = height / 10
    one = np.zeros(width)
    one.fill(255)
    for i in range(1, 10+1):
        img[i * interval][..., 0] = one 
        img[i * interval][..., 1] = one 
        img[i * interval][..., 2] = one 


no = 1
#left_fn = '%s/left/left_%d.png' % (DATAPATH, no)
#right_fn = '%s/right/right_%d.png' % (DATAPATH, no)

left_fn = '%s/left/re2left_%d.png' % (DATAPATH, no)
right_fn = '%s/right/re2right_%d.png' % (DATAPATH, no)

img1 = cv2.imread(left_fn)
img2 = cv2.imread(right_fn)
add_lines(img1)
add_lines(img2)
vis = np.concatenate((img1, img2), axis=1)
cv2.imshow('vis', vis)
cv2.waitKey(0) 
cv2.destroyAllWindows()
