from __future__ import print_function
import numpy as np
import os
import sys
from skimage import io

orignal_list = '../lists/girl_with_ir2'
root_dir = '/data2/virtual3'
target_list = '../lists/exclude_1536.list'

tf = open(target_list, 'w')
with open(orignal_list+'.list', 'r') as f:
    for line in f.readlines():
        r = line.split(' ')
        if len(r) < 4:
            print('error: ', line)
            print(r)
            continue
        leftir = r[0]
        rightir = r[4]
        path = os.path.join(root_dir, leftir)
        img_left = io.imread(path)
        if img_left.shape[0] != 2048:
            print(path)
        else:
            tf.write(line)
tf.close()


