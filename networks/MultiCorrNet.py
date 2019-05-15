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

class MultiCorrNet(nn.Module):

    def __init__(self, ngpu, batchNorm=False, lastRelu=True, input_channel=3):
        super(MultiCorrNet, self).__init__()
        
        self.ngpu = ngpu
        self.batchNorm = batchNorm
        self.input_channel = input_channel

        # shrink and extract features
        self.conv1   = ResBlock(self.input_channel, 32, stride=2) # S/2
        self.conv1_a = ResBlock(32, 32, stride=1) # S/2
        self.conv2   = ResBlock(32, 64, stride=2) # S/4
        self.conv2_a = ResBlock(64, 64, stride=1) # S/4
        self.conv3   = ResBlock(64, 128, stride=2) # S/8
        self.conv3_a = ResBlock(128, 128, stride=1) # S/8
        self.conv4   = ResBlock(128, 256, stride=2) # S/16
        self.conv4_a = ResBlock(256, 256, stride=1) # S/16
        self.conv5   = ResBlock(256, 512, stride=2) # S/32
        self.conv5_a = ResBlock(512, 512, stride=1) # S/32
        self.conv6   = ResBlock(512, 1024, stride=2) # S/64
        self.conv6_a = ResBlock(1024, 1024, stride=1) # S/64

        ## start corr from conv3, output channel is 32 + (max_disp * 2 / 2 + 1) 
        #self.corr1 = corr(64, max_disp=96)
        #self.corr2 = corr(128, max_disp=48)
        #self.corr3 = corr(256, max_disp=24)
        #self.corr4 = corr(512, max_disp=12)
        #self.corr5 = corr(1024, max_disp=6)
        #self.corr6 = corr(2048, max_disp=3)
        self.corr_act = nn.LeakyReLU(0.1, inplace=True)

        # predict flow layer
        self.pred_flow6 = predict_flow(1024 + 3)
        self.pred_flow5 = predict_flow(512)
        self.pred_flow4 = predict_flow(256)
        self.pred_flow3 = predict_flow(128)
        self.pred_flow2 = predict_flow(64)
        self.pred_flow1 = predict_flow(32)
        self.pred_flow0 = predict_flow(16)

        # upscale flow layer
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        # deconv previous layer features
        self.iconv5 = nn.ConvTranspose2d(1032, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(526, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(282, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(178, 64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(162, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(17+self.input_channel, 16, 3, 1, 1)
        
        # deconv concat feature before produce disparity
        self.upconv5 = deconv(1024 + 3, 512)
        self.upconv4 = deconv(512, 256)
        self.upconv3 = deconv(256, 128)
        self.upconv2 = deconv(128, 64)
        self.upconv1 = deconv(64, 32)
        self.upconv0 = deconv(32, 16)

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

        conv1_l = self.conv1(img_left)
        conv2_l = self.conv2(conv1_l)
        conv3_l = self.conv3(conv2_l)
        conv4_l = self.conv4(conv3_l)
        conv5_l = self.conv5(conv4_l)
        conv6_l = self.conv6(conv5_l)

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)
        conv4_r = self.conv4(conv3_r)
        conv5_r = self.conv5(conv4_r)
        conv6_r = self.conv6(conv5_r)

        #out_corr1 = self.corr_act(self.corr1(conv1_l, conv1_r)) # 97
        #out_corr2 = self.corr_act(self.corr2(conv2_l, conv2_r)) # 49
        #out_corr3 = self.corr_act(self.corr3(conv3_l, conv3_r)) # 25
        #out_corr4 = self.corr_act(self.corr4(conv4_l, conv4_r)) # 13
        #out_corr5 = self.corr_act(self.corr5(conv5_l, conv5_r)) # 7
        #out_corr6 = self.corr_act(self.corr6(conv6_l, conv6_r)) # 3
        out_corr1 = self.corr_act(build_corr(conv1_l, conv1_r, 97)) # 97
        out_corr2 = self.corr_act(build_corr(conv2_l, conv2_r, 49)) # 49
        out_corr3 = self.corr_act(build_corr(conv3_l, conv3_r, 25)) # 25
        out_corr4 = self.corr_act(build_corr(conv4_l, conv4_r, 13)) # 13
        out_corr5 = self.corr_act(build_corr(conv5_l, conv5_r, 7)) # 7
        out_corr6 = self.corr_act(build_corr(conv6_l, conv6_r, 3)) # 3

        in_conv6a = torch.cat((self.conv6_a(conv6_l), out_corr6), 1) # 1024 + 3
        in_conv5a = torch.cat((self.conv5_a(conv5_l), out_corr5), 1) # 512 + 7
        in_conv4a = torch.cat((self.conv4_a(conv4_l), out_corr4), 1) # 256 + 13 
        in_conv3a = torch.cat((self.conv3_a(conv3_l), out_corr3), 1) # 128 + 25
        in_conv2a = torch.cat((self.conv2_a(conv2_l), out_corr2), 1) # 64 + 49
        in_conv1a = torch.cat((self.conv1_a(conv1_l), out_corr1), 1) # 32 + 97

        pr6 = self.pred_flow6(in_conv6a)
        upconv5 = self.upconv5(in_conv6a)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, in_conv5a), 1) # 512 + 1 + (512 + 7)
        iconv5 = self.iconv5(concat5)

        pr5 = self.pred_flow5(iconv5)
        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, in_conv4a), 1) # 256 + 1 + (256 + 13)
        iconv4 = self.iconv4(concat4)
        
        pr4 = self.pred_flow4(iconv4)
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, in_conv3a), 1) # 128 + 1 + (128 + 25)
        iconv3 = self.iconv3(concat3)

        pr3 = self.pred_flow3(iconv3)
        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, in_conv2a), 1) # 64 + 1 + (64 + 49)
        iconv2 = self.iconv2(concat2)

        pr2 = self.pred_flow2(iconv2)
        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, in_conv1a), 1) # 32 + 1 + (32 + 97)
        iconv1 = self.iconv1(concat1)

        pr1 = self.pred_flow1(iconv1)
        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1) # 16 + 1 + (3)
        iconv0 = self.iconv0(concat0)

        # predict flow
        pr0 = self.pred_flow0(iconv0)

        # predict flow from dropout output
        # pr6 = self.pred_flow6(F.dropout2d(conv6b))
        # pr5 = self.pred_flow5(F.dropout2d(iconv5))
        # pr4 = self.pred_flow4(F.dropout2d(iconv4))
        # pr3 = self.pred_flow3(F.dropout2d(iconv3))
        # pr2 = self.pred_flow2(F.dropout2d(iconv2))
        # pr1 = self.pred_flow1(F.dropout2d(iconv1))
        # pr0 = self.pred_flow0(F.dropout2d(iconv0))

        # if self.training:
        #     # print("finish forwarding.")
        #     return pr0, pr1, pr2, pr3, pr4, pr5, pr6
        # else:
        #     return pr0

        # can be chosen outside
        return pr0, pr1, pr2, pr3, pr4, pr5, pr6

    def weight_parameters(self):
	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
	return [param for name, param in self.named_parameters() if 'bias' in name]


