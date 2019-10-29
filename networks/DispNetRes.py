from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
#from layers_package.resample2d_package.modules.resample2d import Resample2d
#from layers_package.channelnorm_package.modules.channelnorm import ChannelNorm
#from correlation_package.modules.corr import Correlation1d # from PWC-Net
from networks.submodules import *


class DispNetRes(nn.Module):

    def __init__(self, in_planes, batchNorm=True, lastRelu=False, maxdisp=-1, input_channel=3):
        super(DispNetRes, self).__init__()
        
        self.input_channel = input_channel
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu 
        self.maxdisp = maxdisp
        self.res_scale = 7  # number of residuals

        # improved with shrink res-block layers
        self.conv1   = conv(in_planes, 64, 7, 2, batchNorm=self.batchNorm)
        self.conv2   = ResBlock(64, 128, 2)
        self.conv3   = ResBlock(128, 256, 2)
        self.conv3_1 = ResBlock(256, 256)
        self.conv4   = ResBlock(256, 512, stride=2)
        self.conv4_1 = ResBlock(512, 512)
        self.conv5   = ResBlock(512, 512, stride=2)
        self.conv5_1 = ResBlock(512, 512)
        self.conv6   = ResBlock(512, 1024, stride=2)
        self.conv6_1 = ResBlock(1024, 1024)

        # original shrink conv layers
        #self.conv2   = conv(self.batchNorm, 64, 128, 5, 2)
        #self.conv3   = conv(self.batchNorm, 128, 256, 5, 2)
        #self.conv3_1 = conv(self.batchNorm, 256, 256)
        #self.conv4   = conv(self.batchNorm, 256, 512, stride=2)
        #self.conv4_1 = conv(self.batchNorm, 512, 512)
        #self.conv5   = conv(self.batchNorm, 512, 512, stride=2)
        #self.conv5_1 = conv(self.batchNorm, 512, 512)
        #self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        #self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.pred_res6 = predict_flow(1024)

        # iconv with deconv layers
        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(769, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(385, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(193, 64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(97, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(17+self.input_channel, 16, 3, 1, 1)

        # expand and produce disparity
        self.upconv5 = deconv(1024, 512)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res5 = predict_flow(512)

        self.upconv4 = deconv(512, 256)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res4 = predict_flow(256)

        self.upconv3 = deconv(256, 128)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res3 = predict_flow(128)

        self.upconv2 = deconv(128, 64)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res2 = predict_flow(64)

        self.upconv1 = deconv(64, 32)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res1 = predict_flow(32)

        self.upconv0 = deconv(32, 16)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        if self.maxdisp == -1:
            self.pred_res0 = predict_flow(16)
            self.relu = nn.ReLU(inplace=False) 
        else:
            self.disp_expand = ResBlock(16, self.maxdisp)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, 0.02 / n)
                # m.weight.data.normal_(0, 0.02)
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, inputs, get_feature=False):

        input = inputs[0]
        base_flow = inputs[1]

        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3(conv2)
        conv3b = self.conv3_1(conv3a)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)

        pr6_res = self.pred_res6(conv6b)
        pr6 = pr6_res + base_flow[6]

        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        pr5_res = self.pred_res5(iconv5)
        pr5 = pr5_res + base_flow[5]

        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)

        pr4_res = self.pred_res4(iconv4)
        pr4 = pr4_res + base_flow[4]
        
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3_res = self.pred_res3(iconv3)
        pr3 = pr3_res + base_flow[3]

        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2), 1)
        iconv2 = self.iconv2(concat2)

        pr2_res = self.pred_res2(iconv2)
        pr2 = pr2_res + base_flow[2]

        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1), 1)
        iconv1 = self.iconv1(concat1)

        pr1_res = self.pred_res1(iconv1)
        pr1 = pr1_res + base_flow[1]

        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, input[:, :self.input_channel, :, :]), 1)
        iconv0 = self.iconv0(concat0)

        # predict flow residual
        if self.maxdisp == -1:
            pr0_res = self.pred_res0(iconv0)
            pr0 = pr0_res + base_flow[0]

            if self.lastRelu:
                pr0 = self.relu(pr0)
                pr1 = self.relu(pr1)
                pr2 = self.relu(pr2)
                pr3 = self.relu(pr3)
                pr4 = self.relu(pr4)
                pr5 = self.relu(pr5)
                pr6 = self.relu(pr6)
        else:
            pr0_res = self.disp_expand(iconv0)
            pr0_res = F.softmax(pr0_res, dim=1)
            pr0_res = disparity_regression(pr0_res, self.maxdisp)

        if get_feature:
            return pr0, pr1, pr2, pr3, pr4, pr5, pr6, iconv0
        else:
            return pr0, pr1, pr2, pr3, pr4, pr5, pr6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

