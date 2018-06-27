from __future__ import print_function
import numpy as np
import os
import sys
from skimage import io
from dataset import load_pfm

orignal_list = './lists/virtual02'
#root_dir = '/data2/virtual2'
root_dir = '/'
target_list = './lists/virtual02-1024x1024.list'
tf = open(target_list, 'w')

with open(orignal_list+'.list', 'r') as f:
    for line in f.readlines():
        r = line.strip('\n').split(' ')
        if len(r) < 3:
            continue
        error = False
        for im in r:
            path = os.path.join(root_dir, im)
            img = None
            if not os.path.isfile(path):
                error = True
                break
            if path.find('.png') > 0:
                img = io.imread(path)
            else:
                img, scale = load_pfm(path)
            #print('name: ', im, ', shape: ', img.shape)
            if img.shape[0] != 1024 or img.shape[1] != 1024:
                print(img.shape, line)
                error = True
                break
        if not error:
            tf.write(line)
tf.close()


