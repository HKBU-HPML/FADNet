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
from networks.DispNetC import DispNetC
from networks.NormNetNF import NormNetNF
from networks.submodules import *

class DispNormNet2(nn.Module):

    def __init__(self, batchNorm=False, lastRelu=True, input_channel=3, maxdisp=-1, focus_length={fx=480, fy=480 * 576 / 540}, input_size=(576, 960)):
        super(DispNormNet2, self).__init__()
        self.input_channel = input_channel
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu
        self.maxdisp = maxdisp

        # First Block (DispNetC)
        self.dispnetc = DispNetC(self.batchNorm, input_channel=input_channel, get_features=True)
        # Second and third Block (DispNetS), input is 6+3+1+1=11
        self.normnet = NormNetNF(self.batchNorm, input_channel=3+3+2)

        self.relu = nn.ReLU(inplace=False)

        self.disp2norm = Disp2Norm(batch_size, input_size[1], input_size[0], focus_length)

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
        dispnetc_flows, dispnetc_features = self.dispnetc(inputs)
        dispnetc_flow = dispnetc_flows[0]
        #depthc = dispnetc_flow.clone()
        #depthc[depthc == 0] = 0.01
        #depthc = 48.0 / depthc
        #depthc[depthc > 30] = 30

        norm_input = self.disp2norm.disp2angle(dispnet_flow)

        df5, df4, df3, df2, df1, df0 = depth_features
        nf5 = self.disp2norm.disp2angle(df5)
        nf4 = self.disp2norm.disp2angle(df4)
        nf3 = self.disp2norm.disp2angle(df3)
        nf2 = self.disp2norm.disp2angle(df2)
        nf1 = self.disp2norm.disp2angle(df1)
        nf0 = self.disp2norm.disp2angle(df0)
        normal_features = (nf5, nf4, nf3, nf2, nf1, nf0)


        # normnetdf
        #inputs_normnetdf = torch.cat((inputs, dispnetc_flow), dim = 1)
        #normal = self.normnetdf(inputs_normnetdf, dispnetc_features)

        # normnetnf
        inputs_normnetnf = torch.cat((inputs, norm_input), dim = 1)
        normal = self.normnet(inputs_normnetnf, normal_features)

        normal = normal / torch.norm(normal, 2, dim=1, keepdim=True)

        if self.training:
            return dispnetc_flows, normal
        else:
            return dispnetc_flow, normal# , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    
