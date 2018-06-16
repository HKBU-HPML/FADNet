import numpy as np
import PyEXR as exr
import sys

def load_exr(filename, single_depth=False):
    exrimg = exr.PyEXRImage(filename, single_depth)
    #exrimg = exr.PyEXRImage(filename)
    print('filename:', filename)
    print('height, width: ', exrimg.height, exrimg.width)
    disp_arr = np.array(exrimg, copy = False)
    print('disp shape: ', disp_arr.shape)
    try:
        rgba = np.reshape(disp_arr, (exrimg.height, exrimg.width, 4)) 
    except Exception as e:
        print('Exception: ', filename, e)
        rgba = np.reshape(disp_arr, (exrimg.height, exrimg.width)) 
    rgb = rgba[:,:,0:3]
    rgb = np.flip(rgb, 0)
    print('shape: ', rgb.shape)
    return rgb 


