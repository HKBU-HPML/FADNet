from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
#from correlation_package.modules.corr import Correlation1d # from PWC-Net
from networks.submodules import *


class MobileExtractNet(nn.Module):

    def __init__(self, resBlock=True):
        super(MobileExtractNet, self).__init__()
        
        # shrink and extract features
        self.conv1   = conv(3, 32, 7, 2)
        if resBlock:
            self.conv2   = ResBlock(32, 64, stride=2)
            self.conv3   = ResBlock(64, 128, stride=2)
        else:
            self.conv2   = conv(32, 64, stride=2)
            self.conv3   = conv(64, 128, stride=2)

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

    def forward(self, inputs):

        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        conv1_l = self.conv1(img_left) # 3-32
        conv2_l = self.conv2(conv1_l) # 32-64
        conv3a_l = self.conv3(conv2_l) # 64-128

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3a_r = self.conv3(conv2_r)

        return conv1_l, conv2_l, conv3a_l, conv3a_r

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class MobileDispCUNet(nn.Module):

    def __init__(self, batchNorm=False, lastRelu=True, resBlock=True, maxdisp=-1, input_channel=3):
        super(MobileDispCUNet, self).__init__()
        
        self.batchNorm = batchNorm
        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.relu = nn.ReLU(inplace=False)
        self.corr_max_disp = 40

        # shrink and extract features
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        if resBlock:
            self.conv_redir = ResBlock(128, 32, stride=1)
            self.conv3_1 = ResBlock(72, 64)
            self.conv4   = ResBlock(64, 128, stride=2)
            self.conv5   = ResBlock(128, 256, stride=2)
            self.conv6   = ResBlock(256, 128, stride=2)
        else:
            self.conv_redir = conv(128, 32, stride=1)
            self.conv3_1 = conv(72, 64)
            self.conv4   = conv(64, 128, stride=2)
            self.conv5   = conv(128, 256, stride=2)
            self.conv6   = conv(256, 128, stride=2)

        self.pred_flow6 = predict_flow(128)

        # iconv with deconv
        self.iconv5 = nn.ConvTranspose2d(256+128+1, 128, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(64+128+1, 64, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(64+32+1, 64, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(64+16+1, 32, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(32+16+1, 16, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(17+self.input_channel, 16, 3, 1, 1)

        # expand and produce disparity
        self.upconv5 = deconv(128, 128)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = predict_flow(128)

        self.upconv4 = deconv(128, 64)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = predict_flow(64)

        self.upconv3 = deconv(64, 32)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = predict_flow(64)

        self.upconv2 = deconv(64, 16)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = predict_flow(32)

        self.upconv1 = deconv(32, 16)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = predict_flow(16)

        self.upconv0 = deconv(16, 16)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        if self.maxdisp == -1:
            self.pred_flow0 = predict_flow(16)
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

        #self.freeze()
        
    def forward(self, inputs, conv1_l, conv2_l, conv3a_l, corr_volume):

        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        # Correlate corr3a_l and corr3a_r
        #out_corr = self.corr(conv3a_l, conv3a_r)
        #print('shape: ', conv3a_l.shape)
        out_corr = self.corr_activation(corr_volume)

        out_conv3a_redir = self.conv_redir(conv3a_l) # 128-32
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1) # 32+40

        conv3b = self.conv3_1(in_conv3b) # 72-64
        conv4a = self.conv4(conv3b) # 64-128
        conv5a = self.conv5(conv4a) # 128-256
        conv6a = self.conv6(conv5a) # 256-128

        pr6 = self.pred_flow6(conv6a) # 128-1
        upconv5 = self.upconv5(conv6a) # 128-128
        upflow6 = self.upflow6to5(pr6) # 1-1
        concat5 = torch.cat((upconv5, upflow6, conv5a), 1)
        iconv5 = self.iconv5(concat5) # 128+1+256-128

        pr5 = self.pred_flow5(iconv5) # 128-1
        upconv4 = self.upconv4(iconv5) # 128-64
        upflow5 = self.upflow5to4(pr5) # 1-1
        concat4 = torch.cat((upconv4, upflow5, conv4a), 1)
        iconv4 = self.iconv4(concat4) # 64+1+128-64
        
        pr4 = self.pred_flow4(iconv4) # 64-1
        upconv3 = self.upconv3(iconv4) # 64-32
        upflow4 = self.upflow4to3(pr4) # 1-1
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3) # 32+1+64-64

        pr3 = self.pred_flow3(iconv3) # 64-1
        upconv2 = self.upconv2(iconv3) # 64-16
        upflow3 = self.upflow3to2(pr3) # 1-1
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2) # 16+1+64, 32

        pr2 = self.pred_flow2(iconv2) # 16-1
        upconv1 = self.upconv1(iconv2) # 32-16
        upflow2 = self.upflow2to1(pr2) # 1-1
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1) # 16+1+32-16

        pr1 = self.pred_flow1(iconv1) # 16-1
        upconv0 = self.upconv0(iconv1) # 16-16
        upflow1 = self.upflow1to0(pr1) # 1-1
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0) # 16+1+3-16

        # predict flow
        if self.maxdisp == -1:
            pr0 = self.pred_flow0(iconv0)
            pr0 = self.relu(pr0)
        else:
            pr0 = self.disp_expand(iconv0)
            pr0 = F.softmax(pr0, dim=1)
            pr0 = disparity_regression(pr0, self.maxdisp)

        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)
        return disps
 
    def freeze(self):
        for name, param in self.named_parameters():
            if ('weight' in name) or ('bias' in name):
                param.requires_grad = False

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


