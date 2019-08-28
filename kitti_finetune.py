from __future__ import print_function 
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import logging
from utils.common import load_loss_scheme, logger, formatter
from dataloader import KITTIloader2012 as ls2012
from dataloader import KITTIloader2015 as ls2015
from dataloader import KITTILoader as DA

from networks.DispNetCSRes import DispNetCSRes
from losses.multiscaleloss import multiscaleloss
from losses.balanceloss import MyLoss2

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--loss', type=str, help='indicates the loss scheme', default='simplenet_flying')
args = parser.parse_args()

if not os.path.exists(args.savemodel):
    os.makedirs(args.savemodel)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#if args.datatype == '2015':
#   from dataloader import KITTIloader2015 as ls
#elif args.datatype == '2012':
#   from dataloader import KITTIloader2012 as ls

#all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)
all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls2015.dataloader(args.datapath)
#all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls2012.dataloader("/datasets/kitti2012/training/")
#all_left_12, all_right_12, all_left_disp_12, test_left_12, test_right_12, test_left_disp_12 = ls2012.dataloader("/datasets/kitti2012/training/")

#all_left_img.extend(all_left_12)
#all_right_img.extend(all_right_12)
#all_left_disp.extend(all_left_disp_12)
#test_left_img.extend(test_left_12)
#test_right_img.extend(test_right_12)
#test_left_disp.extend(test_left_disp_12)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 16, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 16, shuffle= False, num_workers= 4, drop_last=False)

#TestImgLoader12 = torch.utils.data.DataLoader(
#         DA.myImageFloder(all_left_12,all_right_12,all_left_disp_12, False), 
#         batch_size= 16, shuffle= False, num_workers= 4, drop_last=False)

devices = [int(item) for item in args.devices.split(',')]
ngpus = len(devices)

if args.model == 'dispnetcres':
    model = DispNetCSRes(ngpus, False, True)
else:
    logger.info('no model')

if args.cuda:
    model = nn.DataParallel(model, device_ids=devices)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), amsgrad=True)

loss_json = load_loss_scheme(args.loss)
train_round = loss_json["round"]
loss_scale = loss_json["loss_scale"]
loss_weights = loss_json["loss_weights"]

myCriterion = MyLoss2(thresh=3, alpha=2)

def train(imgL,imgR,disp_L, criterion):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        #---------
        mask = (disp_true > 0) & (disp_true < 192)
        mask.detach_()
        mask = torch.unsqueeze(mask, 1)
        #----

        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output3,1)
            loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
        elif args.model == 'dispnetcres':
            output_net1, output_net2 = model(torch.cat((imgL, imgR), 1))

            # multi-scale loss
            disp_true = disp_true.unsqueeze(1)
            loss_net1 = criterion(output_net1, disp_true)
            loss_net2 = criterion(output_net2, disp_true)
            #loss = 0.3 * loss_net1 + 0.7 * loss_net2 + myCriterion(output_net2[0][mask], disp_true[mask])
            loss = 0.5 * loss_net1 + loss_net2
            #loss = 0.5 * loss_net1 + myCriterion(output_net2[0][mask], disp_true[mask])

            # only the last scale
            #output1 = output_net1[0].squeeze(1)
            #output2 = output_net2[0].squeeze(1)
            #loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) 

        loss.backward()
        optimizer.step()

        return loss.data.item()

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        #logger.info(imgL.size())
        #imgL = F.pad(imgL, (0, 48, 0, 16), "constant", 0)
        #imgR = F.pad(imgR, (0, 48, 0, 16), "constant", 0)
        #logger.info(imgL.size())

        with torch.no_grad():
            output_net1, output_net2 = model(torch.cat((imgL, imgR), 1))

        pred_disp = output_net2.squeeze(1)
        pred_disp = pred_disp.data.cpu()
        #pred_disp = pred_disp[:, :368, :1232]

        #computing 3-px error#
        true_disp = disp_true.clone()
        index = np.argwhere(true_disp>0)
        small_idx = np.argwhere((true_disp>0)&(true_disp<=25))
        large_idx = np.argwhere(true_disp>25)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        #torch.cuda.empty_cache()
        
        bg_correct = (true_disp[index[0][:], index[1][:], index[2][:]] <= 25) & correct
        fg_correct = (true_disp[index[0][:], index[1][:], index[2][:]] >  25) & correct
        bg_val_err = 1 - (float(torch.sum(bg_correct))/float(len(small_idx[0])))
      