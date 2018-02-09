from __future__ import print_function
from multiprocessing import Process
import os
import argparse
import numpy as np
from skimage import io, transform
from skimage.viewer import ImageViewer
3
from matplotlib import pyplot as plt
from dataset import load_pfm, save_pfm
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976


<<<<<<< Updated upstream
DATAPATH = './data'
OUTPUTPATH = './data'
=======
DATAPATH = '/home/datasets/imagenet'
OUTPUTPATH = '/home/datasets/imagenet'
>>>>>>> Stashed changes
#FILELIST = 'FlyingThings3D_release_TRAIN.list'

def deltaE(rgb_a, rgb_b):
    rgb_object_a = sRGBColor(rgb_a[0], rgb_a[1], rgb_a[2])
    rgb_object_b = sRGBColor(rgb_b[0], rgb_b[1], rgb_b[2])
    xyz_a = convert_color(rgb_object_a, XYZColor, target_illuminant='d50')
    xyz_b = convert_color(rgb_object_b, XYZColor, target_illuminant='d50')
    lab_a = convert_color(xyz_a, LabColor) 
    lab_b = convert_color(xyz_b, LabColor) 
    delta_e = delta_e_cie1976(lab_a, lab_b)
    return delta_e


def remove(left, right, left_disp):
    height = left.shape[0]
    width = left.shape[1]
    print('left shape: ', left.shape)
    for i in range(height):
        for j in range(width):
            depth = int(left_disp[i,j])
            if j-depth<0:
                left_disp[i,j] = 0
                continue
            print("{} {} {}".format(i, j, j - depth))
            left_rgb = left[i, j-depth]
            right_rgb = right[i,j]
            #print('[{}-{}]'.format(left_rgb, right_rgb))
            #print('deltaE: ', deltaE(left_rgb, right_rgb))
            #distance = deltaE(left_rgb, right_rgb)
            distance = np.linalg.norm(left_rgb-right_rgb, 2) #deltaE(left_rgb, right_rgb)
            if distance > 400:
                left_disp[i,j] = 0
    return left_disp

def remove_gt(left, right, left_disp):
    height = left.shape[0]
    width = left.shape[1]
    new_disp = np.array(left, copy=True)
    #print('left shape: ', left.shape)
    for i in range(height):
        for j in range(width):
            depth = int(left_disp[i,j])
<<<<<<< Updated upstream
            #if j-depth<0 or depth >= width or depth<0:
            if j-depth<0 or depth >= width or depth<0:
=======
            if j-depth<0 or depth >= width or depth < 0:
>>>>>>> Stashed changes
                new_disp[i,j] = 0
                continue
            #print("{} {} {}".format(i, j, depth))
            left_rgb = left[i, j]
            right_rgb = right[i,j-depth]
            #print('[{}-{}]'.format(left_rgb, right_rgb))
            #print('deltaE: ', deltaE(left_rgb, right_rgb))
            #distance = deltaE(left_rgb, right_rgb)
            distance = abs(left_rgb - right_rgb) 
            if distance > 2: 
                new_disp[i,j] = 0
    return new_disp 


def clean(img_pairs):
    if not os.path.exists(OUTPUTPATH):
        os.mkdir(OUTPUTPATH)
    total_count = len(img_pairs)
    for i, f in enumerate(img_pairs):
        names = f.split()
        name = names[2]
<<<<<<< Updated upstream
        save_name = os.path.join(OUTPUTPATH, name)
        save_name = save_name.replace('disparity', 'clean_disparity')
=======
        save_name = os.path.join(OUTPUTPATH, name).replace("disparity", "clean_disparity")
>>>>>>> Stashed changes
        save_path = os.path.dirname(save_name)
        if os.path.isfile(save_name):
            continue
        img_left_name = os.path.join(DATAPATH, names[0])
        img_right_name = os.path.join(DATAPATH, names[1])
        #if not '0013' in img_left_name:
        #    continue
        img_left = io.imread(img_left_name)
        img_right = io.imread(img_right_name)
        print('Name: ', name)
        gt_disp_name = os.path.join(DATAPATH, name)
        right_dt_name = gt_disp_name.replace('left', 'right')
        gt_disp, scale = load_pfm(gt_disp_name)
        gt_disp_right, scale = load_pfm(right_dt_name)
        #print('Shape: ', gt_disp.shape)
        #removed_gt_disp = remove(img_left, img_right, gt_disp)
        removed_gt_disp = remove_gt(gt_disp, gt_disp_right, gt_disp)
        #removed_gt_disp = np.flip(removed_gt_disp, axis=0)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_pfm(save_name, removed_gt_disp)
        print('%d out of %d finished' % (i, total_count))
        #plt.imshow(removed_gt_disp, cmap='gray')
        #viewer = ImageViewer(gt_disp)
        #viewer.show()
        #break
    #plt.show()


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
                    for i in range(wanted_parts) ]

def run(filelist, nworkers):
    img_pairs = []
    with open(filelist, "r") as f:
        img_pairs = f.readlines()
    p_img_pairs = split_list(img_pairs, nworkers) 
    print('nworkers: ', nworkers)
    processes = []
    for i in range(nworkers):
        p = Process(target=clean, args=(p_img_pairs[i],))
        processes.append(p)
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
<<<<<<< Updated upstream
    parser.add_argument('--filelist', type=str, help='file list', default='FlyingThings3D_release_TEST.list')
=======
    parser.add_argument('--filelist', type=str, help='file list', default='FlyingThings3D_release_TRAIN.list')
>>>>>>> Stashed changes
    parser.add_argument('--nworkers', type=int, help='number of processes', default=20)
    opt = parser.parse_args()
    #clean(opt.filelist) 
    run(opt.filelist, opt.nworkers)
