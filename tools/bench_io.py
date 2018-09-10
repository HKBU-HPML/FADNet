from __future__ import print_function
import os
import time
import numpy as np
from dataset import load_pfm
from skimage import io
root_dir = '/data2/virtual3'
target_root_dir = '/data3/virtual3npy'
target_list = './lists/exclude_1536.list'

def generate_npy(filelist):
    with open(filelist, 'r') as f:
        counter = 0
        path = None
        counter = 0
        for line in f.readlines():
            imagefiles = line.split(' ')
            for imagefile in imagefiles:
                imagefile = imagefile.strip()
                path = os.path.join(root_dir, imagefile)
                img = None
                tarfn = None
                if imagefile.find('.png') > 0:
                    img = io.imread(path)
                    tarfn = os.path.join(target_root_dir, imagefile.replace('png', 'npy'))
                elif imagefile.find('.pfm') > 0:
                    img, scale = load_pfm(path)
                    tarfn = os.path.join(target_root_dir, imagefile.replace('pfm', 'npy'))
                if img is not None and tarfn:
                    dirname = os.path.dirname(tarfn)
                    #target_dir = dirname.replace('virtual3', 'virtual3npy')
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    if os.path.exists(tarfn):
                        continue
                    print('tarfn: ', tarfn)
                    np.save(tarfn, img)
                    counter += 1
        print('Png converted counter: ', counter)


def bench_read_file(fn):
    if fn.find('.png') > 0:
        img = io.imread(fn)
    elif fn.find('.npy') > 0:
        img = np.load(fn)
    return img

def test_readtime(filelist):
    with open(filelist, 'r') as f:
        counter = 0
        path = None
        for line in f.readlines():
            imagefile = line.split(' ')[0]
            path = os.path.join(root_dir, imagefile)
            img = io.imread(path)
            np.save(os.path.join('/data2/tmp', imagefile.replace('png', 'npy')),img)
            print('if: ', imagefile)
            print('path: ', path)
            break
        s = time.time()
        for i in range(0, 20):
            counter += 1
            bench_read_file(path)
        print('Average time: ', (time.time() -s)/counter)

def bench():
    fn = '/data2/virtual3/ep0002/camera1_R/XNCG_ep0002_cam01_rd_lgt.0001.png'
    #fn = '/data2/tmp/ep0002/camera1_R/XNCG_ep0002_cam01_rd_lgt.0001.npy'
    counter = 20
    s = time.time()
    for i in range(counter):
        bench_read_file(fn)
    print('Average time: ', (time.time() -s)/counter)

def compare():
    fn = '/data2/virtual3/ep0002/camera1_R/XNCG_ep0002_cam01_rd_lgt.0001.png'
    fn1 = '/data2/tmp/ep0002/camera1_R/XNCG_ep0002_cam01_rd_lgt.0001.npy'
    img = io.imread(fn).astype(np.float32)
    img1 = np.load(fn1).astype(np.float32)
    diff = img - img1
    mean = np.mean(diff)
    std = np.std(diff)
    print('mean: ', mean)
    print('std: ', std)




if __name__ == '__main__':
    #bench()
    #test_readtime(target_list)
    #compare()
    generate_npy(target_list)


