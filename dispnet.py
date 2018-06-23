from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import *
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from layers_package.layers import ResBlock
from layers_package.layers import ResBlock
from layers_package.correlation_package.modules.correlation import Correlation
from layers_package.resample2d_package.modules.resample2d import Resample2d
from layers_package.channelnorm_package.modules.channelnorm import ChannelNorm
from layers_package.submodules import *


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1)//2, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=False)
    # return ResBlock(in_planes,1,stride=1)

def deconv(in_planes, out_planes):
    return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
    )

class DispNet(nn.Module):

    def __init__(self, ngpu, batchNorm=True):
        super(DispNet, self).__init__()
        
        self.ngpu = ngpu
        self.batchNorm = batchNorm

        # shrink and extract features
        self.conv1   = conv(self.batchNorm, 6, 64, 7, 2)
        self.conv2   = ResBlock(64, 128, 2)
        self.conv3   = ResBlock(128, 256, 2)
        self.conv3_1 = ResBlock(256, 256)
        self.conv4   = ResBlock(256, 512, stride=2)
        self.conv4_1 = ResBlock(512, 512)
        self.conv5   = ResBlock(512, 512, stride=2)
        self.conv5_1 = ResBlock(512, 512)
        self.conv6   = ResBlock(512, 1024, stride=2)
        self.conv6_1 = ResBlock(1024, 1024)

        #self.conv2   = conv(self.batchNorm, 64, 128, 5, 2)
        #self.conv3   = conv(self.batchNorm, 128, 256, 5, 2)
        #self.conv3_1 = conv(self.batchNorm, 256, 256)
        #self.conv4   = conv(self.batchNorm, 256, 512, stride=2)
        #self.conv4_1 = conv(self.batchNorm, 512, 512)
        #self.conv5   = conv(self.batchNorm, 512, 512, stride=2)
        #self.conv5_1 = conv(self.batchNorm, 512, 512)
        #self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        #self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.pred_flow6 = predict_flow(1024)

        # expand and produce disparity
        self.upconv5 = deconv(1024, 512)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.pred_flow5 = predict_flow(512)

        self.upconv4 = deconv(512, 256)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv4 = nn.ConvTranspose2d(769, 256, 3, 1, 1)
        self.pred_flow4 = predict_flow(256)

        self.upconv3 = deconv(256, 128)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv3 = nn.ConvTranspose2d(385, 128, 3, 1, 1)
        self.pred_flow3 = predict_flow(128)

        self.upconv2 = deconv(128, 64)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv2 = nn.ConvTranspose2d(193, 64, 3, 1, 1)
        self.pred_flow2 = predict_flow(64)

        self.upconv1 = deconv(64, 32)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv1 = nn.ConvTranspose2d(97, 32, 3, 1, 1)
        self.pred_flow1 = predict_flow(32)

        self.upconv0 = deconv(32, 16)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.iconv0 = nn.ConvTranspose2d(20, 16, 3, 1, 1)
        self.pred_flow0 = predict_flow(16)


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
        
    def forward(self, input):

        # split left image and right image
        # print(input.size())
        # img_left = torch.index_select(input, 1, torch.cuda.LongTensor([0, 1, 2]))
        # img_right = torch.index_select(input, 1, torch.cuda.LongTensor[3, 4, 5])
        imgs = torch.chunk(input, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

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

        pr6 = self.pred_flow6(conv6b)

        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)
        pr5 = self.pred_flow5(iconv5)

        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)
        pr4 = self.pred_flow4(iconv4)
        
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)
        pr3 = self.pred_flow3(iconv3)

        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2), 1)
        iconv2 = self.iconv2(concat2)
        pr2 = self.pred_flow2(iconv2)

        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1), 1)
        iconv1 = self.iconv1(concat1)
        pr1 = self.pred_flow1(iconv1)

        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)
        pr0 = self.pred_flow0(iconv0)

	# img_right_rec = warp(img_left, pr0)

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


class DispNetC(nn.Module):

<<<<<<< HEAD
    def __init__(self, ngpu, batchNorm=False, input_channel=3):
=======
    def __init__(self, ngpu, batchNorm=False):
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
        super(DispNetC, self).__init__()
        
        self.ngpu = ngpu
        self.batchNorm = batchNorm
<<<<<<< HEAD
        self.input_channel = input_channel

        # shrink and extract features
        self.conv1   = conv(self.batchNorm, self.input_channel, 64, 7, 2)
=======

        # shrink and extract features
        self.conv1   = conv(self.batchNorm, 3, 64, 7, 2)
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
        self.conv2   = ResBlock(64, 128, 2)
        self.conv3   = ResBlock(128, 256, 2)

	# start corr from conv3, output channel is 32 + 21*21 = 473
	self.conv_redir = ResBlock(256, 32, stride=1)
	self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
	self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        self.conv3_1 = ResBlock(473, 256)
        self.conv4   = ResBlock(256, 512, stride=2)
        self.conv4_1 = ResBlock(512, 512)
        self.conv5   = ResBlock(512, 512, stride=2)
        self.conv5_1 = ResBlock(512, 512)
        self.conv6   = ResBlock(512, 1024, stride=2)
        self.conv6_1 = ResBlock(1024, 1024)

        #self.conv3_1 = conv(self.batchNorm, 256, 256)
        #self.conv4   = conv(self.batchNorm, 256, 512, stride=2)
        #self.conv4_1 = conv(self.batchNorm, 512, 512)
        #self.conv5   = conv(self.batchNorm, 512, 512, stride=2)
        #self.conv5_1 = conv(self.batchNorm, 512, 512)
        #self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        #self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.pred_flow6 = predict_flow(1024)

        # # iconv with resblock
        # self.iconv5 = ResBlock(1025, 512, 1)
        # self.iconv4 = ResBlock(769, 256, 1)
        # self.iconv3 = ResBlock(385, 128, 1)
        # self.iconv2 = ResBlock(193, 64, 1)
        # self.iconv1 = ResBlock(97, 32, 1)
        # self.iconv0 = ResBlock(20, 16, 1)

        # iconv with deconv
        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(769, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(385, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(193, 64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(97, 32, 3, 1, 1)
<<<<<<< HEAD
        self.iconv0 = nn.ConvTranspose2d(17+self.input_channel, 16, 3, 1, 1)
=======
        self.iconv0 = nn.ConvTranspose2d(20, 16, 3, 1, 1)
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3

        # # original iconv with conv
        # self.iconv5 = conv(self.batchNorm, 1025, 512, 3, 1)
        # self.iconv4 = conv(self.batchNorm, 769, 256, 3, 1)
        # self.iconv3 = conv(self.batchNorm, 385, 128, 3, 1)
        # self.iconv2 = conv(self.batchNorm, 193, 64, 3, 1)
        # self.iconv1 = conv(self.batchNorm, 97, 32, 3, 1)
        # self.iconv0 = conv(self.batchNorm, 20, 16, 3, 1)
        
        # expand and produce disparity
        self.upconv5 = deconv(1024, 512)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = predict_flow(512)

        self.upconv4 = deconv(512, 256)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = predict_flow(256)

        self.upconv3 = deconv(256, 128)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = predict_flow(128)

        self.upconv2 = deconv(128, 64)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = predict_flow(64)

        self.upconv1 = deconv(64, 32)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = predict_flow(32)

        self.upconv0 = deconv(32, 16)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow0 = predict_flow(16)


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
        
    def forward(self, input):

        # split left image and right image
        imgs = torch.chunk(input, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        conv1_l = self.conv1(img_left)
        conv2_l = self.conv2(conv1_l)
        conv3a_l = self.conv3(conv2_l)

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3a_r = self.conv3(conv2_r)

        # Correlate corr3a_l and corr3a_r
        out_corr = self.corr(conv3a_l, conv3a_r)
        out_corr = self.corr_activation(out_corr)
	out_conv3a_redir = self.conv_redir(conv3a_l)
	in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

        conv3b = self.conv3_1(in_conv3b)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)


        pr6 = self.pred_flow6(conv6b)
        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        pr5 = self.pred_flow5(iconv5)
        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)
        
        pr4 = self.pred_flow4(iconv4)
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3 = self.pred_flow3(iconv3)
        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)

        pr2 = self.pred_flow2(iconv2)
        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)

        pr1 = self.pred_flow1(iconv1)
        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
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


class DispNetRes(nn.Module):

<<<<<<< HEAD
    def __init__(self, ngpu, in_planes, batchNorm=True, lastRelu=False, input_channel=3):
        super(DispNetRes, self).__init__()
        
        self.ngpu = ngpu
        self.input_channel = input_channel
=======
    def __init__(self, ngpu, in_planes, batchNorm=True, lastRelu=False):
        super(DispNetRes, self).__init__()
        
        self.ngpu = ngpu
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu 
        self.res_scale = 7  # number of residuals

        # improved with shrink res-block layers
<<<<<<< HEAD
        in_planes = input_channel * 3 + 2
=======
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
        self.conv1   = conv(self.batchNorm, in_planes, 64, 7, 2)
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
<<<<<<< HEAD
        self.iconv0 = nn.ConvTranspose2d(17+self.input_channel, 16, 3, 1, 1)
=======
        self.iconv0 = nn.ConvTranspose2d(20, 16, 3, 1, 1)
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3

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
        self.pred_res0 = predict_flow(16)

        self.relu = nn.ReLU(inplace=False) 

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
<<<<<<< HEAD
        concat0 = torch.cat((upconv0, upflow1, input[:, :self.input_channel, :, :]), 1)
=======
        concat0 = torch.cat((upconv0, upflow1, input[:, :3, :, :]), 1)
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
        iconv0 = self.iconv0(concat0)

        # predict flow residual
        pr0_res = self.pred_res0(iconv0)
        pr0 = pr0_res + base_flow[0]

        # # predict flow residual with dropout output
        # pr6_res = self.pred_res6(F.dropout2d(conv6b))
        # pr5_res = self.pred_res5(F.dropout2d(iconv5))
        # pr4_res = self.pred_res4(F.dropout2d(iconv4))
        # pr3_res = self.pred_res3(F.dropout2d(iconv3))
        # pr2_res = self.pred_res2(F.dropout2d(iconv2))
        # pr1_res = self.pred_res1(F.dropout2d(iconv1))
        # pr0_res = self.pred_res0(F.dropout2d(iconv0))

        if self.lastRelu:
            if get_feature:
<<<<<<< HEAD
                return self.relu(pr0), self.relu(pr1), self.relu(pr2), self.relu(pr3), self.relu(pr4), self.relu(pr5), self.relu(pr6), iconv1
=======
                return self.relu(pr0), self.relu(pr1), self.relu(pr2), self.relu(pr3), self.relu(pr4), self.relu(pr5), self.relu(pr6), iconv0
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
            else:
                return self.relu(pr0), self.relu(pr1), self.relu(pr2), self.relu(pr3), self.relu(pr4), self.relu(pr5), self.relu(pr6)
        if get_feature:
            return pr0, pr1, pr2, pr3, pr4, pr5, pr6, iconv0

        return pr0, pr1, pr2, pr3, pr4, pr5, pr6

    def weight_parameters(self):
	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
	return [param for name, param in self.named_parameters() if 'bias' in name]

class DispNetCSRes(nn.Module):

<<<<<<< HEAD
    def __init__(self, ngpus, batchNorm=True, lastRelu=False, input_channel=3):
        super(DispNetCSRes, self).__init__()
        self.input_channel = input_channel
=======
    def __init__(self, ngpus, batchNorm=True, lastRelu=False):
        super(DispNetCSRes, self).__init__()
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu

        # First Block (DispNetC)
<<<<<<< HEAD
        self.dispnetc = DispNetC(ngpus, self.batchNorm, input_channel=input_channel)
=======
        self.dispnetc = DispNetC(ngpus, self.batchNorm)
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3

        # warp layer and channelnorm layer
        self.channelnorm = ChannelNorm()
        self.resample1 = Resample2d()

        # Second Block (DispNetRes), input is 11 channels(img0, img1, img1->img0, flow, diff-mag)
<<<<<<< HEAD
        self.dispnetres = DispNetRes(ngpus, self.input_channel, self.batchNorm, lastRelu=self.lastRelu, input_channel=input_channel)

=======
        self.dispnetres = DispNetRes(ngpus, 11, self.batchNorm, lastRelu=self.lastRelu)
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
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
        dummy_flow = torch.autograd.Variable(torch.zeros(dispnetc_final_flow.data.shape).cuda())
        # dispnetc_final_flow_2d = torch.cat((target, dummy_flow), dim = 1)
        dispnetc_final_flow_2d = torch.cat((dispnetc_final_flow, dummy_flow), dim = 1)
<<<<<<< HEAD
        resampled_img1 = self.resample1(inputs[:, self.input_channel:, :, :], -dispnetc_final_flow_2d)
        diff_img0 = inputs[:, :self.input_channel, :, :] - resampled_img1
=======
        resampled_img1 = self.resample1(inputs[:, 3:, :, :], -dispnetc_final_flow_2d)
        diff_img0 = inputs[:, :3, :, :] - resampled_img1
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        inputs_net2 = torch.cat((inputs, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim = 1)

        # dispnetres
        dispnetres_flows = self.dispnetres([inputs_net2, dispnetc_flows])
        dispnetres_final_flow = dispnetres_flows[0]
        

        if self.training:
            return dispnetc_flows, dispnetres_flows
        else:
            return dispnetc_final_flow, dispnetres_final_flow# , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1


    def weight_parameters(self):
	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
	return [param for name, param in self.named_parameters() if 'bias' in name]



class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DispNetCSResWithDomainTransfer(nn.Module):

    def __init__(self, ngpus, batchNorm=True, lastRelu=False):
        super(DispNetCSResWithDomainTransfer, self).__init__()
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu

        # First Block (DispNetC)
        self.dispnetc = DispNetC(ngpus, self.batchNorm)

        # warp layer and channelnorm layer
        self.channelnorm = ChannelNorm()
        self.resample1 = Resample2d()

        # Second Block (DispNetRes), input is 11 channels(img0, img1, img1->img0, flow, diff-mag)
        self.dispnetres = DispNetRes(ngpus, 11, self.batchNorm, lastRelu=self.lastRelu)
        self.relu = nn.ReLU(inplace=False)

        self.domain_classifier = nn.Sequential()
<<<<<<< HEAD
        #self.domain_classifier.add_module('d_fc1', nn.Linear(512*512*2, 100))
        self.domain_classifier.add_module('d_fc1', nn.Linear(256*256*32, 50))
        self.domain_classifier.add_module('d_sigmoid', nn.Sigmoid())
        self.domain_classifier.add_module('d_fc2', nn.Linear(50, 2))
=======
        self.domain_classifier.add_module('d_fc1', nn.Linear(512*512*2, 100))
        self.domain_classifier.add_module('d_sigmoid', nn.Sigmoid())
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())


    def forward(self, inputs, alpha):

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
        dummy_flow = torch.autograd.Variable(torch.zeros(dispnetc_final_flow.data.shape).cuda())
        # dispnetc_final_flow_2d = torch.cat((target, dummy_flow), dim = 1)
        dispnetc_final_flow_2d = torch.cat((dispnetc_final_flow, dummy_flow), dim = 1)
        resampled_img1 = self.resample1(inputs[:, 3:, :, :], -dispnetc_final_flow_2d)
        diff_img0 = inputs[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        inputs_net2 = torch.cat((inputs, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim = 1)

        # dispnetres
        dispnetres_flows = self.dispnetres([inputs_net2, dispnetc_flows], get_feature=True)
        dispnetres_final_flow = dispnetres_flows[0]
<<<<<<< HEAD
        #feature = dispnetc_final_flow_2d.view(-1, 512*512*2)
        #print('size; ', dispnetres_flows[-1].size())
        feature = dispnetres_flows[-1].view(-1, 256*256*32)
=======
        feature = dispnetc_final_flow_2d.view(-1, 512*512*2)
>>>>>>> 188781b4360dfcda5534e3782d360e4ae400c8c3
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        if self.training:
            return dispnetc_flows, dispnetres_flows[0:-1], domain_output
        else:
            return dispnetc_final_flow, dispnetres_final_flow# , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1


    def weight_parameters(self):
	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
	return [param for name, param in self.named_parameters() if 'bias' in name]

