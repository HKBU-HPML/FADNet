import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10

import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from time import time
from struct import unpack
import matplotlib.pyplot as plt
import re
import numpy as np
import pdb
from path import Path

from utils.preprocess import scale_disp, default_transform
from networks.FADNet import FADNet

parser = argparse.ArgumentParser(description='FADNet')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--sceneflow', type=int, default=0, help='sceneflow dataset? Default=False')
parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--middlebury', type=int, default=0, help='Middlebury? Default=False')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='data path')
parser.add_argument('--list', default='lists/middeval_test.list',
                    help='list of stereo images')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--savepath', default='results/',
                    help='path to save the results.')
parser.add_argument('--model', default='fadnet',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
opt = parser.parse_args()
print(opt)

torch.backends.cudnn.benchmark = True

opt.cuda = not opt.no_cuda and torch.cuda.is_available()

if not os.path.exists(opt.savepath):
    os.makedirs(opt.savepath)

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

if opt.model == 'psmnet':
    model = PSMNet(opt.maxdisp)
elif opt.model == 'fadnet':
    model = FADNet(False, True)
else:
    print('no model')
    sys.exit(-1)

if opt.loadmodel is not None:
    state_dict = torch.load(opt.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def readPFM(file): 
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)

    return img, height, width

def save_pfm(filename, image, scale=1):
    '''
    Save a Numpy array to a PFM file.
    '''
    color = None
    file = open(filename, "w")
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)

def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width: 
        # padding zero 
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp    
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]	
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def load_data_imn(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    h, w, c = np.shape(left)

    rgb_transform = default_transform()
    img_left = rgb_transform(left)
    img_right = rgb_transform(right)

    bottom_pad = 1024-h
    right_pad = 1536-w
    img_left = np.lib.pad(img_left,((0,0),(0,bottom_pad),(0,right_pad)),mode='constant',constant_values=0)
    img_right = np.lib.pad(img_right,((0,0),(0,bottom_pad),(0,right_pad)),mode='constant',constant_values=0)
    return torch.from_numpy(img_left).float(), torch.from_numpy(img_right).float(), h, w

def test_md(leftname, rightname, savename, imgname):

    input1, input2, height, width = load_data_imn(leftname, rightname)

    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if opt.cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
        input_var = torch.cat((input1, input2), 0)
        input_var = input_var.unsqueeze(0)
    torch.cuda.synchronize()
    start_time = time()
    with torch.no_grad():
        prediction = model(input_var)[1]
        prediction = prediction.squeeze(0)
    torch.cuda.synchronize()
    end_time = time()
    
    print("Processing time: {:.4f}".format(end_time - start_time))
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    temp = temp[0, :height, :width]
    savepfm_path = savename.replace('.png','') 
    temp = np.flipud(temp)

    disppath = Path(savepfm_path)
    disppath.makedirs_p()
    save_pfm(savepfm_path+'/disp0FADNet.pfm', temp, scale=1)
    ##########write time txt########
    fp = open(savepfm_path+'/timeFADNet.txt', 'w')
    runtime = "%.4f" % (end_time - start_time)  
    fp.write(runtime)   
    fp.close()

def test_kitti(leftname, rightname, savename):
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)
 
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():        
        prediction = model(input1, input2)
        
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    skimage.io.imsave(savename, (temp * 256).astype('uint16'))


def test(leftname, rightname, savename):  
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()

    start_time = time()
    with torch.no_grad():
        prediction = model(input1, input2)
    end_time = time()
    
    print("Processing time: {:.4f}".format(end_time - start_time))
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height or width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    plot_disparity(savename, temp, 192)
    savename_pfm = savename.replace('png','pfm') 
    temp = np.flipud(temp)

def plot_disparity(savename, data, max_disp):
    plt.imsave(savename, data, vmin=0, vmax=max_disp, cmap='turbo')

   
if __name__ == "__main__":
    file_path = opt.datapath
    file_list = opt.list
    f = open(file_list, 'r')
    filelist = f.readlines()
    for index in range(len(filelist)):
        current_file = filelist[index].split()
        if opt.kitti2015:
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
            savename = opt.savepath + current_file[0: len(current_file) - 1]
            test_kitti(leftname, rightname, savename)

        if opt.kitti2012:
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
            savename = opt.savepath + current_file[0: len(current_file) - 1]
            test_kitti(leftname, rightname, savename)

        if opt.sceneflow:
            leftname = file_path + 'frames_finalpass/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'frames_finalpass/' + current_file[0: len(current_file) - 14] + 'right/' + current_file[len(current_file) - 9:len(current_file) - 1]
            leftgtname = file_path + 'disparity/' + current_file[0: len(current_file) - 4] + 'pfm'
            disp_left_gt, height, width = readPFM(leftgtname)
            savenamegt = opt.savepath + "{:d}_gt.png".format(index)
            plot_disparity(savenamegt, disp_left_gt, 192)

            savename = opt.savepath + "{:d}.png".format(index)
            test(leftname, rightname, savename)

        if opt.middlebury:
            leftname = file_path + current_file[0]
            rightname = file_path + current_file[1]

            temppath = opt.savepath.replace(opt.savepath.split("/")[-2], opt.savepath.split("/")[-2]+"/images")     
            img_path = Path(temppath)
            img_path.makedirs_p()
            savename = opt.savepath + leftname[0: len(leftname) - 8] + ".png"
            img_name = img_path + leftname[0: len(leftname) - 8] + ".png"
            print(img_name, savename)
            test_md(leftname, rightname, savename, img_name)

