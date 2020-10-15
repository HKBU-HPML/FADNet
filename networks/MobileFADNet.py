from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from layers_package.resample2d_package.resample2d import Resample2d
from layers_package.channelnorm_package.channelnorm import ChannelNorm
from networks.MobileDispNetC import MobileDispNetC
from networks.MobileDispNetRes import MobileDispNetRes
from networks.submodules import *

class MobileFADNet(nn.Module):

    def __init__(self, batchNorm=True, lastRelu=False, resBlock=True, maxdisp=-1, input_channel=3, input_img_shape=None, warp_size=None):
        super(MobileFADNet, self).__init__()
        self.input_channel = input_channel
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu
        self.maxdisp = maxdisp
        self.resBlock = resBlock
        self.warp_size = warp_size #(1, 3, 576, 960)
        if warp_size is not None:
            B, C, H, W = warp_size
            xx = torch.arange(0, W).cuda().float()
            yy = torch.arange(0, H).cuda().float()
            xx = xx.view(1,-1).repeat(H,1)
            yy = yy.view(-1,1).repeat(1,W)
            xx = xx.view(1,1,H,W).repeat(B,1,1,1)
            yy = yy.view(1,1,H,W).repeat(B,1,1,1)
            #grid = torch.cat((xx,yy),1).float()
            self.warp_grid = (xx, yy)
        else:
            self.warp_grid = None

        # First Block (DispNetC)
        self.dispnetc = MobileDispNetC(self.batchNorm, maxdisp=self.maxdisp, input_channel=input_channel, input_img_shape=input_img_shape)

        # warp layer and channelnorm layer
        self.channelnorm = ChannelNorm()
        self.resample1 = Resample2d()

        # Second Block (DispNetRes), input is 11 channels(img0, img1, img1->img0, flow, diff-mag)
        in_planes = 3 * 3 + 1 + 1
        self.dispnetres = MobileDispNetRes(in_planes, self.batchNorm, lastRelu=self.lastRelu, maxdisp=self.maxdisp, input_channel=input_channel)

        self.relu = nn.ReLU(inplace=False)

        # # parameter initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if m.bias is not None:
        #             init.uniform(m.bias)
        #         init.xavier_uniform(m.weight)

        #     if isinstance(m, nn.ConvTranspose2d):
        #         if m.bias is not None:
        #             init.uniform(m.bias)
        #         init.xavier_uniform(m.weight)

    def forward(self, inputs):

        # split left image and right image
        # inputs = inputs_target[0]
        # target = inputs_target[1]
        imgs = torch.chunk(inputs, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        # dispnetc
        dispnetc_flows = self.dispnetc(inputs)
        dispnetc_final_flow = dispnetc_flows[0]

        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        #dummy_flow = torch.autograd.Variable(torch.zeros(dispnetc_final_flow.data.shape).cuda())
        #dispnetc_final_flow_2d = torch.cat((dispnetc_final_flow, dummy_flow), dim = 1)
        #resampled_img1 = self.resample1(inputs[:, self.input_channel:, :, :], -dispnetc_final_flow_2d)
        #norm_diff_img0 = self.channelnorm(diff_img0)
        resampled_img1 = warp_right_to_left(inputs[:, self.input_channel:, :, :], -dispnetc_final_flow, warp_grid=self.warp_grid)
        diff_img0 = inputs[:, :self.input_channel, :, :] - resampled_img1
        norm_diff_img0 = channel_length(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        inputs_net2 = torch.cat((inputs, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim = 1)

        # dispnetres
        dispnetres_flows = self.dispnetres([inputs_net2, dispnetc_flows])
        index = 0
        #print('Index: ', index)
        dispnetres_final_flow = dispnetres_flows[index]
        

        if self.training:
            return dispnetc_flows, dispnetres_flows
        else:
            return dispnetc_final_flow, dispnetres_final_flow# , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1


    def weight_parameters(self):
    	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
    	return [param for name, param in self.named_parameters() if 'bias' in name]


