from __future__ import print_function
from numpy import linalg as LA
from readers import load_exr
from dataset import load_pfm, save_pfm
import numpy as np
import os
import cv2
import scipy.misc
import PyEXR as exr

OUTPUTPATH = '/media/sf_Shared_Data/dispnet/cam01-pfm/'
#OUTPUTPATH = '/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl/'

#FOCAL_LENGTH = 100.0 # 35mm
#PPI = 90.0/25.4
#FOCAL_LENGTH *= PPI

FOV = 37.849197 
FOCAL_LENGTH = (1024*0.5) / np.tan(FOV* 0.5 * np.pi/180)
print('Focal length: ', FOCAL_LENGTH)

BASELINE = 15/10. # 15cm = 150mm

def depth_to_disparity(focal_length, baseline, depth):
    return focal_length * baseline/ (depth) #* 1024.0

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    return image ** invGamma

def transform_exr_to_pfm(filename, path, outputpath):
    """
    1. Save RGB to PNG
    2. Transform depth to disparity, save as pfm
    """
    single_depth = filename.find('.Z.') > 0
    if single_depth:
        dst_filename = filename.replace('exr', 'pfm')
    else:
        dst_filename = filename.replace('exr', 'png')
    absfn = os.path.join(path, filename)
    absdstfn = os.path.join(outputpath, dst_filename)
    if os.path.exists(absdstfn):
        return
    depth_arr = load_exr(absfn, single_depth)
    if single_depth:
        disp_arr = depth_to_disparity(FOCAL_LENGTH, BASELINE, depth_arr)
    else:
        disp_arr = depth_arr
    print('srcfilename: %s', absfn)
    if single_depth:
        disp_arr = disp_arr[:,:,0]
        save_pfm(absdstfn, disp_arr)
    else:
        min_val = np.min(disp_arr)
        max_val = np.max(disp_arr)
        bgr = np.zeros(disp_arr.shape, dtype=np.float32)
        bgr[:,:,0]=disp_arr[:,:,2]
        bgr[:,:,1]=disp_arr[:,:,1]
        bgr[:,:,2]=disp_arr[:,:,0]
        #for i in range(0,3):
        #bgr[:,:,i]= (bgr[:,:,i]-min_val[i])/ (max_val[i]-min_val[i]) * 255.
        #bgr= (bgr-min_val) / (max_val-min_val) * 255.
        bgr= bgr * 255.
        bgr[bgr>255]=255.0
        bgr = adjust_gamma(bgr, 0.9)
        bgr[bgr>255]=255.0
        bgr = np.flip(bgr, 0) 
        #cv2.imwrite(absdstfn, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(absdstfn, bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        print('max: ', np.max(bgr))
        #scipy.misc.imsave(absdstfn, disp_arr)

def validate_disparity(dispfile, leftfile, rightfile, path):
    outputfilename = os.path.join(OUTPUTPATH, 'val.'+dispfile+'.pfm')
    resfilename = os.path.join(OUTPUTPATH, 'res.'+dispfile+'.pfm')
    depthfilename = os.path.join(OUTPUTPATH, 'depth.'+dispfile+'.pfm')
    #dispfile = os.path.join(OUTPUTPATH, dispfile)
    dispfile = os.path.join(path, dispfile)
    leftfile = os.path.join(path, leftfile)
    rightfile = os.path.join(path, rightfile)
    left = load_exr(leftfile, False)
    right = load_exr(rightfile, False)
    depth_arr = load_exr(dispfile, True)
    disp = depth_to_disparity(FOCAL_LENGTH, BASELINE, depth_arr)
    #disp+=60
    #disp, _ = load_pfm(dispfile)
    shape = disp.shape
    print('--dispshape: ', shape)
    val_right = np.zeros(right.shape, dtype=left.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            d = int(disp[i][j][0])
            #print('disp: ', d)
            right_j = j-d
            if right_j < 0 or right_j >= shape[1]:
                continue
            val_right[i][right_j] = left[i][j]
    res = val_right - right
    save_pfm(depthfilename, depth_arr)
    save_pfm(outputfilename, val_right)
    save_pfm(resfilename, np.abs(res))
    print('res norm: ', LA.norm(res))
    print('left to right norm: ', LA.norm(left-right))


def process_exrs_to_pfms(filelist, path, outputpath):
    f = open(filelist)
    for line in f.readlines():
        #if '0040' not in line:
        #    continue
        line = line.rstrip()
        single_depth = line.find('.Z.') > 0
        #if not single_depth:
        transform_exr_to_pfm(line, path, outputpath)
        if not single_depth:
            #Transform left image
            line = line.replace('R', 'L')
            transform_exr_to_pfm(line, path, outputpath)


if __name__ == '__main__':
    path = '/media/sf_Shared_Data/dispnet/cam01/'
    #filename = 'girl_camera1_Lcamera1_L.Z.0166.exr'
    #transform_exr_to_pfm(filename, path, OUTPUTPATH)
    #filename = 'girl_camera1_Lcamera1_L.0166.exr'
    filelist = 'exrfilelistR.txt'
    process_exrs_to_pfms(filelist, path, OUTPUTPATH)

    #leftfile = 'girl_camera1_Rcamera1_R.0246.exr'
    #rightfile = 'girl_camera1_Lcamera1_L.0246.exr'
    #dispfile = 'girl_camera1_Rcamera1_R.Z.0246.exr'
    #validate_disparity(dispfile, leftfile, rightfile, path)
