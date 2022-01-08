from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
#from layers_package.resample2d_package.resample2d import Resample2d
#from layers_package.channelnorm_package.channelnorm import ChannelNorm
from networks.DispNetC import DispNetC
from networks.DispNetS import DispNetS
from networks.submodules import *

class DispNetCS(nn.Module):

    def __init__(self, batchNorm=False, lastRelu=True, resBlock=True, maxdisp=-1, input_channel=3):
        super(DispNetCS, self).__init__()
        self.input_channel = input_channel
        self.batchNorm = batchNorm
        self.resBlock = resBlock
        self.lastRelu = lastRelu
        self.maxdisp = maxdisp

        # First Block (DispNetC)
        self.dispnetc = DispNetC(batchNorm=self.batchNorm, resBlock=self.resBlock, maxdisp=self.maxdisp, input_channel=input_channel)
        # Second and third Block (DispNetS), input is 6+3+1+1=11
        self.dispnets1 = DispNetS(11, batchNorm=self.batchNorm, resBlock=self.resBlock, maxdisp=self.maxdisp, input_channel=3)

        # warp layer and channelnorm layer
        self.channelnorm = ChannelNorm()
        self.resample1 = Resample2d()

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
        dispnetc_flow = dispnetc_flows[0]

        # dispnets1
        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        dummy_flow_1 = torch.autograd.Variable(torch.zeros(dispnetc_flow.data.shape).cuda())
        # dispnetc_final_flow_2d = torch.cat((target, dummy_flow), dim = 1)
        dispnetc_flow_2d = torch.cat((dispnetc_flow, dummy_flow_1), dim = 1)
        resampled_img1_1 = self.resample1(inputs[:, self.input_channel:, :, :], -dispnetc_flow_2d)
        diff_img0_1 = inputs[:, :self.input_channel, :, :] - resampled_img1_1
        norm_diff_img0_1 = self.channelnorm(diff_img0_1)

        # concat img0, img1, img1->img0, flow, diff-mag
        inputs_net2 = torch.cat((inputs, resampled_img1_1, dispnetc_flow, norm_diff_img0_1), dim = 1)
        dispnets1_flows = self.dispnets1(inputs_net2)
        dispnets1_flow = dispnets1_flows[0]
        

        if self.training:
            return dispnetc_flows, dispnets1_flows
        else:
            return dispnetc_flow, dispnets1_flow # , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


