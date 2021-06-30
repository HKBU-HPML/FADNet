import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from skimage import io
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
from networks.DispNetC import DispNetC
from networks.GANet_deep import GANet
from networks.stackhourglass import PSMNet

parser = argparse.ArgumentParser(description='FADNet')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--sceneflow', type=int, default=0, help='sceneflow dataset? Default=False')
parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--middlebury', type=int, default=0, help='Middlebury? Default=False')
parser.add_argument('--eth3d', type=int, default=0, help='ETH3D? Default=False')
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
elif opt.model == 'ganet':
    model = GANet(opt.maxdisp)
elif opt.model == 'fadnet':
    model = FADNet(maxdisp=opt.maxdisp)
elif opt.model == 'dispnetc':
    model = DispNetC(resBlock=False, maxdisp=opt.maxdisp)
elif opt.model == 'crl':
    model = FADNet(resBlock=False, maxdisp=opt.maxdisp)
else:
    print('no model')
    sys.exit(-1)

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if opt.loadmodel is not None:
    state_dict = torch.load(opt.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

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
    #left = Image.open(leftname)
    #right = Image.open(rightname)
    #h, w, c = np.shape(left)
    left = io.imread(leftname)
    right = io.imread(rightname)
    h, w, _ = left.shape

    normalize = 'instnorm'
    if normalize == 'imagenet':
        rgb_transform = default_transform()
        img_left = rgb_transform(left)
        img_right = rgb_transform(right)
    else:
        img_left = np.zeros([3, h, w], 'float32')
        img_right = np.zeros([3, h, w], 'float32')
        for c in range(3):
            img_left[c, :, :] = (left[:, :, c] - np.mean(left[:, :, c])) / np.std(left[:, :, c])
            img_right[c, :, :] = (right[:, :, c] - np.mean(right[:, :, c])) / np.std(right[:, :, c])

    print(h, w)
    bottom_pad = opt.crop_height-h
    right_pad = opt.crop_width-w
    img_left = np.lib.pad(img_left,((0,0),(0,bottom_pad),(0,right_pad)),mode='constant',constant_values=0)
    img_right = np.lib.pad(img_right,((0,0),(0,bottom_pad),(0,right_pad)),mode='constant',constant_values=0)
    return torch.from_numpy(img_left).float(), torch.from_numpy(img_right).float(), h, w

def test_md(leftname, rightname, savename):

    print(savename)
    epe = 0
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

    # print epe
    thres = 400
    if 'trainingQ' in leftname:
        thres = 192
    if 'training' in leftname:
        gt_disp, _, _ = readPFM(leftname.replace('im0.png', 'disp0GT.pfm'))
        gt_disp[np.isinf(gt_disp)] = 0
        mask = (gt_disp > 0) & (gt_disp < thres)
        epe = np.mean(np.abs(gt_disp[mask] - temp[mask])) 
        print(savename, epe, np.min(gt_disp), np.max(gt_disp))

    savepfm_path = savename.replace('.png','') 
    temp = np.flipud(temp)

    disppath = Path(savepfm_path)
    disppath.makedirs_p()
    save_pfm(savepfm_path+'/disp0FADNet++.pfm', temp, scale=1)
    ##########write time txt########
    fp = open(savepfm_path+'/timeFADNet++.txt', 'w')
    runtime = "%.4f" % (end_time - start_time)  
    fp.write(runtime)   
    fp.close()

    return epe

def test_eth3d(leftname, rightname, savename):

    print(savename)
    epe = 0
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

    # print epe
    if 'training' in leftname:
        gt_disp, _, _ = readPFM(leftname.replace('im0.png', 'disp0GT.pfm').replace('training/', 'training_gt/'))
        gt_disp[np.isinf(gt_disp)] = 0
        mask = (gt_disp > 0) & (gt_disp < 192)
        epe = np.mean(np.abs(gt_disp[mask] - temp[mask])) 
        print(savename, epe, np.min(gt_disp), np.max(gt_disp))

    temp = np.flipud(temp)

    disppath = Path('/'.join(savename.split('/')[:-1]))
    disppath.makedirs_p()
    save_pfm(savename, temp, scale=1)
    ##########write time txt########
    fp = open(savename.replace("pfm", "txt"), 'w')
    runtime = "runtime %.4f" % (end_time - start_time)  
    fp.write(runtime)   
    fp.close()

    return epe

def test_kitti(leftname, rightname, savename):
    print(savename)
    epe = 0
    input1, input2, height, width = load_data_imn(leftname, rightname)

    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if opt.cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
        input_var = torch.cat((input1, input2), 0)
        input_var = input_var.unsqueeze(0)
    with torch.no_grad():        
        prediction = model(input_var)[1]
        prediction = prediction.squeeze(0)
        
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    temp = temp[0, :height, :width]

    # print epe
    if 'training' in leftname:
        gt_disp = Image.open(leftname.replace('colored_0', 'disp_occ').replace('image_2', 'disp_occ_0'))
        gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256
        mask = (gt_disp > 0) & (gt_disp < 192)
        epe = np.mean(np.abs(gt_disp[mask] - temp[mask])) 
        print(savename, epe, np.min(gt_disp), np.max(gt_disp))

    skimage.io.imsave(savename, (temp * 256).astype('uint16'))

    return epe


def test(leftname, rightname, savename, gt_disp):  
    input1, input2, height, width = load_data_imn(leftname, rightname)

    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    start_time = time()

    if opt.cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
        input_var = torch.cat((input1, input2), 0)
        input_var = input_var.unsqueeze(0)
    with torch.no_grad():
        predictions = model(input_var)
        if len(predictions) > 2:
            prediction = predictions[0]
        elif len(predictions) == 2:
            prediction = predictions[1]
        else:
            prediction = predictions
        if prediction.dim() == 4:
            prediction = prediction.squeeze(0)

    end_time = time()
    print("Processing time: {:.4f}".format(end_time - start_time))

    temp = prediction.cpu()
    temp = temp.detach().numpy()
    temp = temp[0, :height, :width]

    plot_disparity(savename, temp, np.max(gt_disp)+5, cmap='rainbow')

    mask = (gt_disp > 0) & (gt_disp < 192)
    epe = np.mean(np.abs(gt_disp[mask] - temp[mask])) 
    err_map = np.abs(temp - gt_disp)
    err_name = savename.replace('disp', 'err')
    plot_disparity(err_name, err_map, 30, cmap='turbo')

    savename_pfm = savename.replace('png','pfm') 
    temp = np.flipud(temp)

    return epe

def plot_disparity(savename, data, max_disp, cmap='turbo'):
    plt.imsave(savename, data, vmin=0, vmax=max_disp, cmap=cmap)

   
if __name__ == "__main__":
    file_path = opt.datapath
    file_list = opt.list
    f = open(file_list, 'r')
    filelist = f.readlines()

    error = 0
    for index in range(len(filelist)):
        current_file = filelist[index].split()
        if opt.kitti2015 or opt.kitti2012:
            leftname = os.path.join(file_path, current_file[0])
            rightname = os.path.join(file_path, current_file[1])
            savename = os.path.join(opt.savepath, current_file[0].split("/")[-1])
            error += test_kitti(leftname, rightname, savename)

        if opt.sceneflow:
            leftname = file_path + current_file[0]
            rightname = file_path + current_file[1] 
            leftgtname = file_path + current_file[2]
            disp_left_gt, height, width = readPFM(leftgtname)
            savenamegt = opt.savepath + "gt_" + "_".join(current_file[2].split("/")[-4:]).replace('pfm', 'png').replace('left', 'disp')
            plot_disparity(savenamegt, disp_left_gt, np.max(disp_left_gt)+5, cmap='rainbow')

            savename = opt.savepath + "%s_" % opt.model + "_".join(current_file[2].split("/")[-4:]).replace('pfm', 'png').replace('left', 'disp')
            epe = test(leftname, rightname, savename, disp_left_gt)
            error += epe
            print(leftname, rightname, savename, epe)

        if opt.middlebury:
            leftname = file_path + current_file[0]
            rightname = file_path + current_file[1]

            temppath = opt.savepath.replace(opt.savepath.split("/")[-2], opt.savepath.split("/")[-2]+"/images")     
            #img_path = Path(temppath)
            #img_path.makedirs_p()
            savename = opt.savepath + "/".join(leftname.split("/")[-4:-1]) + ".png"
            error += test_md(leftname, rightname, savename)
        if opt.eth3d:
            leftname = file_path + current_file[0]
            rightname = file_path + current_file[1]
            savename = opt.savepath + "low_res_two_view/" + leftname.split("/")[-2] + ".pfm"
            error += test_eth3d(leftname, rightname, savename)

        if index > 200:
            break
    print("EPE:", error / len(filelist))

