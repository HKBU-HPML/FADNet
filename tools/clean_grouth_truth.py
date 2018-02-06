from __future__ import print_function
import os
import numpy as np
from skimage import io, transform
from skimage.viewer import ImageViewer
3
from matplotlib import pyplot as plt
from dataset import load_pfm
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976


DATAPATH = '/media/sf_Shared_Data/gpuhomedataset'
OUTPUTPATH = './tmp'
FILELIST = 'FlyingThings3D_release_TRAIN.list'

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
    print('left shape: ', left.shape)
    for i in range(height):
        for j in range(width):
            depth = int(left_disp[i,j])
            if j-depth<0 or depth >= width:
                new_disp[i,j] = 0
                continue
            left_rgb = left[i, j]
            right_rgb = right[i,j-depth]
            #print('[{}-{}]'.format(left_rgb, right_rgb))
            #print('deltaE: ', deltaE(left_rgb, right_rgb))
            #distance = deltaE(left_rgb, right_rgb)
            distance = abs(left_rgb - right_rgb) 
            if distance > 3: 
                new_disp[i,j] = 0
    return new_disp 


def clean(file_list):
    img_pairs = []
    with open(file_list, "r") as f:
        img_pairs = f.readlines()
    for f in img_pairs:
        names = f.split()
        name = names[2]
        img_left_name = os.path.join(DATAPATH, names[0])
        img_right_name = os.path.join(DATAPATH, names[1])
        if not '0013' in img_left_name:
            continue
        img_left = io.imread(img_left_name)
        img_right = io.imread(img_right_name)
        print('Name: ', name)
        gt_disp_name = os.path.join(DATAPATH, name)
        right_dt_name = gt_disp_name.replace('left', 'right')
        gt_disp, scale = load_pfm(gt_disp_name)
        gt_disp_right, scale = load_pfm(right_dt_name)
        print('Shape: ', gt_disp.shape)
        #removed_gt_disp = remove(img_left, img_right, gt_disp)
        removed_gt_disp = remove_gt(gt_disp, gt_disp_right, gt_disp)
        removed_gt_disp = np.flip(removed_gt_disp, axis=0)
        plt.imshow(removed_gt_disp, cmap='gray')
        #viewer = ImageViewer(gt_disp)
        #viewer.show()
        break
    plt.show()


if __name__ == '__main__':
    clean(FILELIST) 
