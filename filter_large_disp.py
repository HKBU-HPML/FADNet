from utils.preprocess import *
import numpy as np
from PIL import Image, ImageOps
import os

def load_disp(filename):
    gt_disp = None
    if filename.endswith('pfm'):
        gt_disp, scale = load_pfm(filename)
        gt_disp = gt_disp[::-1, :]
    elif filename.endswith('npy'):
        gt_disp = np.load(filename)
        gt_disp = gt_disp[::-1, :]
    elif filename.endswith('exr'):
        gt_disp = load_exr(filename)
    else:
        gt_disp = Image.open(filename)
        gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256

    gt_disp[np.isinf(gt_disp)] = 0
    return gt_disp

filelist = "lists/all_files.list"
datapath = "data"
thres = 250

imgns = []
disp_Lns = []
with open(filelist, 'r') as f:
    contents = f.readlines()
    imgns = contents
    disp_Lns = [os.path.join(datapath, line.split()[-1]) for line in contents]

md_amount = 0
eth_amount = 0
kitti_amount = 0
flying_amount = 0
driving_amount = 0
monkaa_amount = 0
ld_imgns = []
for imgn, disp_Ln in zip(imgns, disp_Lns):
    disp_L = load_disp(disp_Ln)
    if np.max(disp_L) > thres:
        print(imgn)
        ld_imgns.append(imgn)

        if "MiddEval" in imgn:
            md_amount += 1
        if "eth3d" in imgn:
            eth_amount += 1
        if "kitti" in imgn:
            kitti_amount += 1
        if "FlyingThings" in imgn:
            flying_amount += 1
        if "driving" in imgn:
            driving_amount += 1
        if "monkaa" in imgn:
            monkaa_amount += 1

print("MiddEval: %d" % md_amount)
print("ETH3D: %d" % eth_amount)
print("KITTI: %d" % kitti_amount)
print("Flying: %d" % flying_amount)
print("Driving: %d" % driving_amount)
print("Monkaa: %d" % monkaa_amount)
print(len(ld_imgns))

ld_list = "lists/large_disp.list"
with open(ld_list, "w") as f:
    for ld_imgn in ld_imgns:
        f.write(ld_imgn)
