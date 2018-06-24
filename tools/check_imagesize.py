from __future__ import print_function
import numpy as np
import os
import sys
from skimage import io
from dataset import load_pfm

orignal_list = './lists/girl_with_ir2'
root_dir = '/data2/virtual3'
target_list = './lists/exclude_1536.list'
tf = open(target_list, 'w')

with open(orignal_list+'.list', 'r') as f:
    for line in f.readlines():
        r = line.strip('\n').split(' ')
        if len(r) < 4:
            print('error: ', line)
            print(r)
            continue
        error = False
        for im in r:
            path = os.path.join(root_dir, im)
            img = None
            if path.find('.png') > 0:
                img = io.imread(path)
            else:
                img, scale = load_pfm(path)
            #print('name: ', im, ', shape: ', img.shape)
            if img.shape[0] != 2048 or img.shape[1] != 2048:
                print('error img: ', img.shape, line, im)
                error = True
                break
        if not error:
            tf.write(line)
tf.close()


