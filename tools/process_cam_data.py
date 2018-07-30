from __future__ import print_function
import cv2
from numpy import linalg as LA
from readers import load_exr
from dataset import load_pfm, save_pfm
import numpy as np
import os
import disp_to_depth as dd
import scipy.misc
import gc
import PyEXR as exr

OUTPUTPATH = '/media/sf_Shared_Data/dispnet/cam01-pfm/'
#OUTPUTPATH = '/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl/'

#FOCAL_LENGTH = 100.0 # 35mm
#PPI = 90.0/25.4
#FOCAL_LENGTH *= PPI

FOV = 37.849197 
FOCAL_LENGTH = (1024*0.5) / np.tan(FOV* 0.5 * np.pi/180)
#print('Focal length: ', FOCAL_LENGTH)
def _focal_length(width):
    return (width*0.5) / np.tan(FOV* 0.5 * np.pi/180)

BASELINE = 15./10. # 15cm = 150mm

def depth_to_disparity(focal_length, baseline, depth):
    return focal_length * baseline/ (depth) #* 1024.0

def disparity_to_depth(focal_length, baseline, disp):
    return focal_length * baseline/ (disp) #* 1024.0

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
    if path:
        outputfilename = os.path.join(OUTPUTPATH, 'val.'+dispfile+'.pfm')
        resfilename = os.path.join(OUTPUTPATH, 'res.'+dispfile+'.pfm')
        depthfilename = os.path.join(OUTPUTPATH, 'depth.'+dispfile+'.pfm')
        dispfile = os.path.join(path, dispfile)
        leftfile = os.path.join(path, leftfile)
        rightfile = os.path.join(path, rightfile)
    else:
        outputfilename = os.path.join(OUTPUTPATH, 'val.'+os.path.basename(dispfile)+'.pfm')
        resfilename = os.path.join(OUTPUTPATH, 'res.'+os.path.basename(dispfile)+'.pfm')
        depthfilename = os.path.join(OUTPUTPATH, 'depth.'+os.path.basename(dispfile)+'.pfm')
    if leftfile.find('.exr') > 0:
        left = load_exr(leftfile, False)
        right = load_exr(rightfile, False)
    else:
        left = cv2.imread(leftfile).astype(np.float32)
        left = np.flip(left, 0)
        right = cv2.imread(rightfile).astype(np.float32)
        right = np.flip(right, 0)
    print('dispfile: ', dispfile)
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

def generate_filelist(path):
    if not os.path.isdir(path):
        return
    leftlist = []
    rightlist = []
    displist = []
    rightdianzhenlist = []
    leftdianzhenlist = []
    for root, dirs, files in os.walk(path):
        #root_path = root.split(os.sep)
        #print('path: ', root)
        for filename in files:
            if filename.find('dianzhen') > 0:
                fullpath = os.path.join(root, filename)
                if fullpath.find('L') > 0:
                    leftdianzhenlist.append(fullpath)
                else:
                    rightdianzhenlist.append(fullpath)
                continue
            if (filename.find('.png') > 0 or filename.find('.exr') > 0) and not filename.find('dianzhen') > 0 and not filename.find('.lock') > 0:
                fullpath = os.path.join(root, filename)
                #print(fullpath) if fullpath.find('Z')>0 and fullpath.find('R') > 0 else print
                if fullpath.find('.exr') > 0 and fullpath.find('Z') > 0:
                    dst_file = fullpath.replace('.exr', '.pfm')
                    #if os.path.exists(dst_file):
                    #    continue
                    #depth_arr = load_exr(fullpath, True)
                    ##disp_arr = depth_to_disparity(FOCAL_LENGTH, BASELINE, depth_arr)
                    #width=depth_arr.shape[0]
                    ##print('width: ', width)
                    #disp_arr = depth_to_disparity(_focal_length(width), BASELINE, depth_arr)
                    #disp_arr = disp_arr[:,:,0]
                    #save_pfm(dst_file, disp_arr)

                    fullpath = dst_file
                else:
                    pass
                    #print(fullpath)
                if filename.find('.png') > 0:
                    if fullpath.find('L') > 0:
                        #print(fullpath)
                        leftlist.append(fullpath)
                    elif fullpath.find('R') > 0:
                        #print(fullpath)
                        rightlist.append(fullpath)
                elif fullpath.find('R') > 0 and fullpath.find('Z') > 0:
                    #print(fullpath)
                    displist.append(fullpath)
                else:
                    #print('fuck:', fullpath)
                    pass
    leftlist.sort()
    rightlist.sort()
    displist.sort()
    leftdianzhenlist.sort()
    rightdianzhenlist.sort()
    #gc.collect()
    #print('len right:',len(rightlist))
    #print('len left:', len(leftlist))
    #print('len disp;', len(displist))
    #print('len dianzhen;', len(dianzhenlist))
    #if  len(rightlist) != len(leftlist) or len(leftlist) != len(displist):
    #    print(rightlist[0], leftlist[0], displist[0])
    #    print(rightlist[-1], leftlist[-1], displist[-1])
    #    print('len: ', len(rightlist),  len(leftlist), len(displist))
    #    raise
    for i in range(0, len(leftlist)):
        #if len(leftdianzhenlist) > 0:
        #    print(rightlist[i], leftlist[i], displist[i], rightdianzhenlist[i], leftdianzhenlist[i])
        print(rightlist[i], leftlist[i], displist[i])

def convert_single_channel_to_multi_channel(path, exrfile):
    import PyEXR as exr
    fn = os.path.join(path, exrfile)
    outfn = os.path.join(path, 'p'+exrfile)
    print('outfn: ', outfn)
    disp2depth = exrfile.find('.pfm') > 0
    if disp2depth:
        disp, s = load_pfm(fn)
        depth = disparity_to_depth(FOCAL_LENGTH, BASELINE, disp)
        print('shape: ', disp.shape)
        print('type: ', disp.dtype)
        outfn = outfn.replace('.pfm', '.exr')
        dd.save_openexr(depth, outfn)
        #save_pfm(outfn, depth)
    else:
        exrimg = exr.PyEXRImage(fn, True)
        exrimg.save(outfn)


if __name__ == '__main__':
    #path = '/media/sf_Shared_Data/dispnet/cam01/'
    #filename = 'girl_camera1_Lcamera1_L.Z.0166.exr'
    #transform_exr_to_pfm(filename, path, OUTPUTPATH)
    #filename = 'girl_camera1_Lcamera1_L.0166.exr'
    #filelist = 'exrfilelistR.txt'
    #process_exrs_to_pfms(filelist, path, OUTPUTPATH)

    #for i in ['girl05', 'girl0011',  'girl0012',  'girl06',  'girl07',  'girl08']:
    #for i in ['ep0010', 'ep0013','ep0014','ep0015','ep0016','ep0017','ep0019','ep0020']:
    for i in range(1, 41): 
        #if i == 32 or i == 36:
        #    continue
        fn = 'ep00'+str(i) if i >= 10 else 'ep000'+str(i)
        #path = '/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl05'
        #path = '/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/%s'% i
        path = '/data2/virtual/%s'% fn 
        generate_filelist(path)

    #path = '/media/sf_Shared_Data/dispnet/ep001/'
    #leftfile = 'girl_camera1_Rcamera1_R.0246.exr'
    #rightfile = 'girl_camera1_Lcamera1_L.0246.exr'
    #dispfile = 'girl_camera1_Rcamera1_R.Z.0246.exr'
    #validate_disparity(dispfile, leftfile, rightfile, path)
#
    #leftfile = '/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl03/R/camera1_R/XNCG_ep0002_cam01_rd_lgt.0051.png'
    #rightfile ='/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl03/L/camera1_L/XNCG_ep0002_cam01_rd_lgt.0051.png'
    #dispfile = '/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl03/R/camera1_R/XNCG_ep0002_cam01_rd_lgt.Z.0051.exr'
    #validate_disparity(dispfile, leftfile, rightfile, None)

    #for cam in range(0, 8):
    #    #cam = 0
    #    path = '/media/sf_Shared_Data/dispnet/FusionPortal/data/%d/' % cam
    #    print('path: ', path)
    #    #convert_single_channel_to_multi_channel(path, '0.exr')
    #    convert_single_channel_to_multi_channel(path, '0.pfm')

