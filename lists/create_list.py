from __future__ import print_function
import numpy as np
import os
import sys

orignal_list = 'exclude_1536'
train_list = orignal_list+'_TRAIN.list'
test_list = orignal_list+'_TEST.list'

trainf = open(train_list, 'w')
testf = open(test_list, 'w')
with open(orignal_list+'.list', 'r') as f:
    for line in f.readlines():
        r = np.random.randint(0, 100)
        if r < 10: # for test
            testf.write(line)
        else:
            trainf.write(line)

trainf.close()
testf.close()



