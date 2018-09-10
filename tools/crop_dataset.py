import numpy as np
from skimage import io
from dataset import *
import os

src_dir = "/home/vradmin/data/virtual3/"
dest_dir = "/data3/virtual3-1024/"

f = open("lists/exclude_1536.list")
imgList = f.readlines()[5727:]
f.close()

i = 0
for line in imgList:
    print line
    img_left, img_right, disp_left, ir_left, ir_right = line.split()

    np_img_left = io.imread(src_dir + img_left)
    np_img_right = io.imread(src_dir + img_right)
    np_disp_left, scale = load_pfm(src_dir + disp_left)
    np_ir_left = io.imread(src_dir + ir_left)
    np_ir_right = io.imread(src_dir + ir_right)

    crop_np_img_left = np_img_left[512:1536, 512:1536, :]
    crop_np_img_right = np_img_right[512:1536, 512:1536, :]
    crop_np_disp_left = np_disp_left[512:1536, 512:1536]

    dest_sub_dir = dest_dir + os.path.dirname(img_left)

    if not os.path.exists(dest_sub_dir):
        os.system("mkdir -p %s" % dest_sub_dir)

    dest_sub_dir = dest_dir + os.path.dirname(img_right)

    if not os.path.exists(dest_sub_dir):
        os.system("mkdir -p %s" % dest_sub_dir)

    io.imsave(os.path.join(dest_dir, img_left), crop_np_img_left)
    io.imsave(os.path.join(dest_dir, img_right), crop_np_img_right)
    save_pfm(os.path.join(dest_dir, disp_left), crop_np_disp_left)

    i += 1

    if i % 100 == 0:
        print i
