from __future__ import print_function
import os
import numpy as np
import csv
from matplotlib import pyplot as plt
from dataset import load_pfm

DATAPATH = '/media/sf_Shared_Data/gpuhomedataset'
OUTPUTPATH = './tmp'
#FILELIST = 'FlyingThings3D_release_TEST.list'
FILELIST = 'FlyingThings3D_release_TRAIN.list'
RESULTLIST = 'NEW_' + FILELIST
CLEANRESULTLIST = 'CLEAN_' + FILELIST

def plot_hist(d, save=False, filename=None, plot=True):
    flatten = d.ravel()
    mean = np.mean(flatten)
    max = np.max(flatten)
    std = np.std(flatten)
    print('len: %d, mean: %.3f, std: %.3f' % (len(flatten), mean, std))
    #return n_neg, flatten.size # return #negative, total
    if plot:
        #count, bins, ignored = plt.hist(flatten, 50, normed=True)
        count, bins, ignored = plt.hist(flatten, bins=np.arange(0,300))
        if save:
            plt.savefig(os.path.join(OUTPUTPATH, '%s.png'%filename), bbox_inches='tight')
        else:
            plt.show()
        plt.clf()
    return mean, std, max


def statistic(file_list):
    img_pairs = []
    with open(file_list, "r") as f:
        img_pairs = f.readlines()
    csv_file = open(RESULTLIST, 'a')
    for f in img_pairs:
        names = f.split()
        name = names[2]
        print('Name: ', name)
        gt_disp_name = os.path.join(DATAPATH, name)
        gt_disp, scale = load_pfm(gt_disp_name)
        print('Shape: ', gt_disp.shape, ', Mean: ', np.mean(gt_disp))

        name_items = name.split('/')
        save_name = 'hist_{}_{}_{}'.format(name_items[-4], name_items[-3], name_items[-1].split('.')[0])
        mean, std, max = plot_hist(gt_disp, save=True, filename=save_name, plot=False)

        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerow([name, mean, std, max])
    csv_file.close()

def statistic_with_file(fn):
    result_file = open(CLEANRESULTLIST, 'a')
    with open(fn, 'r') as f:
        total_array = []
        fns = []
        for line in f.readlines():
            items = line.split('\t')
            total_array.append([float(i) for i in items[1:]])
            fns.append(items[0])
        total_array = np.array(total_array)
        print('Shape: ', total_array[:, 0].shape)
        for i, mean in enumerate(total_array[:, 0]):
            if mean < 150:
                grt = fns[i]
                name_items = grt.split('/')
                left = 'FlyingThings3D_release/frames_cleanpass/%s/%s/%s/left/%s.png' % (name_items[-5], name_items[-4], name_items[-3], name_items[-1].split('.')[0])
                right = 'FlyingThings3D_release/frames_cleanpass/%s/%s/%s/right/%s.png' % (name_items[-5], name_items[-4], name_items[-3], name_items[-1].split('.')[0])
                #result_file.write("%s %s %s\n" % (left, right, fns[i]))

        plot_hist(total_array[:, 0])
        #plot_hist(total_array[:, 1])
        #plot_hist(total_array[:, 2])
    result_file.close()


if __name__ == '__main__':
    #statistic(FILELIST)
    statistic_with_file(RESULTLIST)
