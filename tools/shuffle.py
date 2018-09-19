import os
import random
import math
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, help='file list', default='girl02')
parser.add_argument('--trainPer', type=float, help='percentage of train', default=0.9)

opt = parser.parse_args()

dataset = opt.dataset

list_file = 'lists/%s.list' % dataset
train_list_file = 'lists/%s_TRAIN.list' % dataset
test_list_file = 'lists/%s_TEST.list' % dataset
f = open(list_file, 'r')
recs = f.readlines()
f.close()

random.shuffle(recs)

totalNum = len(recs)
trainNum = int(math.floor(totalNum * opt.trainPer))
testNum = totalNum - trainNum

print totalNum, trainNum, testNum

train_f = open(train_list_file, 'w')
train_f.writelines(recs[:trainNum])
train_f.close()
test_f = open(test_list_file, 'w')
test_f.writelines(recs[trainNum:])
test_f.close()



