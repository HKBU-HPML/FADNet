from __future__ import print_function
import os
import numpy as np
import csv
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image, ImageOps
from skimage import io
from utils.preprocess import load_pfm
FONTSIZE=20
matplotlib.rcParams.update({'font.size': FONTSIZE})
#matplotlib.rc('xtick', labelsize=FONTSIZE) 
#matplotlib.rc('ytick', labelsize=FONTSIZE)
#matplotlib.rc('xlabel', labelsize=FONTSIZE) 
#matplotlib.rc('ylabel', labelsize=FONTSIZE)

#DATAPATH = '/media/sf_Shared_Data/gpuhomedataset/dispnet'
#DATAPATH = '/home/datasets/imagenet/dispnet/virtual'
#DATAPATH = '/home/datasets/imagenet/dispnet'
DATAPATH = './data'
OUTPUTPATH = '/tmp'
#FILELIST = 'FlyingThings3D_release_TEST.list'
#FILELIST = 'FlyingThings3D_release_TRAIN.list'
#FILELIST='lists/real_release.list'
#FILELIST='lists/kitti-groundtruth.list'
#FILELIST='lists/kitti2015_train.list'
#FILELIST='lists/MB2014_TRAIN.list'
#FILELIST='lists/eth3d_train.list'
FILELIST='lists/FlyingThings3D_release_TRAIN_100.list'
RESULTLIST = 'NEW_' + FILELIST
CLEANRESULTLIST = 'CLEAN_' + FILELIST

def plot_hist(d, save=False, filename=None, plot=True, color='r'):
    flatten = d.ravel()
    mean = np.mean(flatten)
    max = np.max(flatten)
    std = np.std(flatten)
    print('len: %d, mean: %.3f, std: %.3f' % (len(flatten), mean, std))
    #return n_neg, flatten.size # return #negative, total
    if plot:
        #count, bins, ignored = plt.hist(flatten, 50, color=color)
        flatten = flatten[np.abs(flatten) > 0.0]
        count, bins, ignored = plt.hist(flatten, bins=np.arange(0,300), density=True, color=color)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Disparity')
        plt.ylabel('Percentage')
        if save:
            plt.savefig(os.path.join(OUTPUTPATH, '%s.pdf'%filename), bbox_inches='tight')
        else:
            #plt.show()
            pass
        #plt.clf()
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


def statistic_mean_std(filelist):
    img_pairs = []
    with open(filelist, "r") as f:
        img_pairs = f.readlines()
    means = []
    for f in img_pairs:
        names = f.split()
        leftname = names[0]
        rightname = names[1]
        leftfn = os.path.join(DATAPATH, leftname)
        rightfn = os.path.join(DATAPATH, rightname)
        leftimgdata = io.imread(leftfn)
        rightimgdata = io.imread(rightfn)
        leftmean = np.mean(leftimgdata.ravel())
        rightmean = np.mean(rightimgdata.ravel())
        print('leftmean: ', leftmean)
        print('rightmean: ', rightmean)
        means.append((leftmean+rightmean)/2)
    means = np.array(means)
    print('total mean: ', np.mean(means))
    print('total std: ', np.std(means))


def statistic_disparity(filelist):
    img_pairs = []
    with open(filelist, "r") as f:
        img_pairs = f.readlines()
    all = np.array([], dtype=np.float32)
    for f in img_pairs:
        names = f.split()
        dispname = names[2]
        fn = os.path.join(DATAPATH, dispname)
        print('fn: ', fn)
        if fn.find('.png') >= 0:
            gt_disp = Image.open(fn)
            gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256
        else:
            gt_disp, _ = load_pfm(fn)
            gt_disp[np.isinf(gt_disp)] = 0
        all = np.concatenate((gt_disp.ravel(), all))
    mean = np.mean(all)
    std = np.std(all)
    color='#A9D18E'
    mean, std, max = plot_hist(all, save=True, filename=filelist, plot=True, color=color)
    print('total mean: ', mean)
    print('total std: ', std)

def statistic_kitti_disparity(filelist):
    img_pairs = []
    with open(filelist, "r") as f:
        img_pairs = f.readlines()
    all = np.array([], dtype=np.float32)
    for f in img_pairs:
        dispname = f[:-1]
        fn = dispname 
        gt_disp = Image.open(fn)
        gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256
        #gt_disp, _ = load_pfm(fn)
        all = np.concatenate((gt_disp.ravel(), all))
    np.save('stat.npy', all)
    mean = np.mean(all)
    std = np.std(all)
    print('total mean: ', mean)
    print('total std: ', std)
    mean, std, max = plot_hist(all, save=True, filename='real_disp.png', plot=False, color='r')

def force_plot():
    all = np.load('stat.npy')
    mean, std, max = plot_hist(all, save=True, filename='real_disp.png', plot=True, color='r')

def plot_hist_with_filename(fn):
    fnt='img00000.bmp'
    leftfn = '/media/sf_Shared_Data/gpuhomedataset/dispnet/real_release/frames_cleanpass/left/%s'%fnt
    rightfn = '/media/sf_Shared_Data/gpuhomedataset/dispnet/real_release/frames_cleanpass/right/%s'%fnt
    realimgdata = io.imread(leftfn)
    #leftfn = '/media/sf_Shared_Data/gpuhomedataset/FlyingThings3D_release/frames_cleanpass/TRAIN/A/0001/left/%s'%fn
    #rightfn = '/media/sf_Shared_Data/gpuhomedataset/FlyingThings3D_release/frames_cleanpass/TRAIN/A/0001/right/%s'%fn
    #realimgdata = io.imread(leftfn)

    leftfn = '/media/sf_Shared_Data/gpuhomedataset/FlyingThings3D_release/frames_cleanpass/TRAIN/A/0000/left/%s'%fn
    rightfn = '/media/sf_Shared_Data/gpuhomedataset/FlyingThings3D_release/frames_cleanpass/TRAIN/A/0000/right/%s'%fn
    leftimgdata = io.imread(leftfn)
    rightimgdata = io.imread(rightfn)
    mean, std, max = plot_hist(leftimgdata, save=False, filename=None, plot=True, color='r')
    mean, std, max = plot_hist(realimgdata, save=False, filename=None, plot=True, color='b')
    plt.show()

def extract_exception_of_occulution():
    #occulution_list = 'CC_FlyingThings3D_release_TRAIN.list'
    #occulution_list = './lists/CC_FlyingThings3D_release_TRAIN.list'
    occulution_list = './lists/girl20_TRAIN.list'
    img_pairs = []
    with open(occulution_list, "r") as f:
        img_pairs = f.readlines()
    means = []
    maxcount = 10000
    i = 0
    for f in img_pairs:
        names = f.split()
        name = names[2]
        #gt_disp_name = os.path.join(DATAPATH, 'clean_dispnet', name)
        gt_disp_name = os.path.join(DATAPATH,  name)
        if not os.path.isfile(gt_disp_name):
            #print('Not found: ', gt_disp_name)
            continue
        gt_disp, scale = load_pfm(gt_disp_name)
        mean = np.mean(gt_disp)
        means.append(mean)
        i+=1
        if i > maxcount:
            break
        print('Name: ', name, ', Mean: ', mean, ', std: ', np.std(gt_disp), ' min: ', np.min(gt_disp), ' max: ', np.max(gt_disp))
    np.save('virtualmean.log', np.array(means))
    #mean, std, max = plot_hist(np.array(means), save=False, filename=None, plot=True, color='r')
    #plt.show()


def parse_mean_log():
    filename = './logs/meanstd_test.log'
    f = open(filename, 'r')
    means = []
    fns = []
    for line in f.readlines():
        mean = line.split()[-4]
        means.append(float(mean))
        fns.append(line.split()[1])
    means = np.array(means)
    fns = np.array(fns)
    k = 10
    #sorted = np.argsort(means)[-k:]
    sorted = np.argsort(means)[:k]
    print(sorted)
    print(means[sorted])
    print(fns[sorted])
    #plt.scatter(range(0, len(means)), means)
    #plot_hist(np.array(means), plot=True)
    #plt.show()


if __name__ == '__main__':
    #statistic(FILELIST)
    #statistic_with_file(RESULTLIST)
    #fn='img00000.bmp'
    #fn='0006.png'
    #plot_hist_with_filename(fn)
    #statistic_mean_std(FILELIST)
    statistic_disparity(FILELIST)
    #statistic_kitti_disparity(FILELIST)
    #extract_exception_of_occulution()
    #parse_mean_log()
