from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils.preprocess import scale_disp, default_transform
#from models import *

from networks.DispNetCSRes import DispNetCSRes
from networks.DispNetC import DispNetC
from losses.multiscaleloss import multiscaleloss

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--savepath', default='results/',
                    help='path to save the results.')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  


test_left_img, test_right_img = DA.dataloader(args.datapath)

devices = [int(item) for item in args.devices.split(',')]
ngpus = len(devices)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
elif args.model == 'dispnetcres':
    model = DispNetCSRes(ngpus, False, True)
elif args.model == 'dispnetc':
    model = DispNetC(ngpus, False, True)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=devices)
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR, imgsize):
        model.eval()
        #imgL = imgL.unsqueeze(0)
        #imgR = imgR.unsqueeze(0)

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     
           #imgL = imgL.cuda()
           #imgR = imgR.cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)
        #imgL = F.pad(imgL, (0, 1280 - imgsize[1], 0, 384 - imgsize[0]), "constant", 0)
        #imgR = F.pad(imgR, (0, 1280 - imgsize[1], 0, 384 - imgsize[0]), "constant", 0)

        #print(imgL.size(), imgR.size())
        with torch.no_grad():
            #output = model(imgL,imgR)
            output_net1, output_net2 = model(torch.cat((imgL, imgR), 1))
            #output_net2 = model(torch.cat((imgL, imgR), 1))[0]
        output = torch.squeeze(output_net2)
        #output = output_net2
        pred_disp = output.data.cpu().numpy()

        #pred_disp = pred_disp[:imgsize[0], :imgsize[1]]
        #pred_disp = scale_disp(pred_disp[np.newaxis, :], (1, img_size[0], imgsize[1]))[0]
        #print(pred_disp.shape)

        #pred_disp = np.flip(pred_disp[0], axis=0)

        return pred_disp


def main():
   #processed = preprocess.get_transform(augment=False)

   for inx in range(len(test_left_img)):

       imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
       imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))

       # resize
       imgsize = imgL_o.shape[:2]
       #imgL_o = skimage.transform.resize(imgL_o, (384, 1280), preserve_range=True)
       #imgR_o = skimage.transform.resize(imgR_o, (384, 1280), preserve_range=True)

       #imgL = processed(imgL_o).numpy()
       #imgR = processed(imgR_o).numpy()
       rgb_transform = default_transform()
       imgL = rgb_transform(imgL_o).numpy()
       imgR = rgb_transform(imgR_o).numpy()

       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

       # pad to (384, 1280)
       top_pad = 384-imgL.shape[2]
       left_pad = 1280-imgL.shape[3]
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp = test(imgL,imgR,imgsize)
       print('time = %.2f' %(time.time() - start_time))

       top_pad   = 384-imgL_o.shape[0]
       left_pad  = 1280-imgL_o.shape[1]
       img = pred_disp[top_pad:,:-left_pad]
       #img = pred_disp

       skimage.io.imsave(os.path.join(args.savepath, test_left_img[inx].split('/')[-1]),(img*256).astype('uint16'))

if __name__ == '__main__':
   main()






