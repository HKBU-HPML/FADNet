from __future__ import print_function

import torch
import torch.nn as nn

from torch.nn.init import kaiming_normal
from correlation_package.modules.corr import Correlation1d # from PWC-Net
from networks.submodules import conv, predict_flow, deconv, ResBlock


class DispNetCorr2(nn.Module):

    def __init__(self, batchNorm=False, using_resblock=False):
        super(DispNetCorr2, self).__init__()
        
        self.batchNorm = batchNorm

        # shrink and extract features
        self.conv1   = conv(self.batchNorm, 3, 64, 7, 2) # 384*192
        if using_resblock:
            self.conv2   = ResBlock(64, 128, 2) # 192*96
            self.conv3   = ResBlock(128, 256, 2) # 96*48
            self.conv3a = ResBlock(256, 256, 1) # 96*48
        else:
            self.conv2   = conv(self.batchNorm, 64, 128, 5, 2) # 192*96
            self.conv3   = conv(self.batchNorm, 128, 256, 5, 2) # 96*48
            self.conv3a = conv(self.batchNorm, 256, 256, 1, stride=1) # 96*48

        self.corr3a = Correlation1d(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=1, corr_multiply=1)
        self.corr3a_act = nn.LeakyReLU(0.1, inplace=True)

        if using_resblock:
            self.conv4 = ResBlock(256, 512, 2) # 48*24
            self.conv4a = ResBlock(512, 512, 1) # 48*24
        else:
            self.conv4 = conv(self.batchNorm, 256, 512, 3, 2) # 48*24
            self.conv4a = conv(self.batchNorm, 512, 512, 1, 1) # 48*24

        self.corr4a = Correlation1d(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=1, corr_multiply=1)
        self.corr4a_act = nn.LeakyReLU(0.1, inplace=True)

        if using_resblock:
            self.corr3a_conv1 = ResBlock(41, 128, 2) # 48*24
            self.corr3a_conv2 = ResBlock(128, 256, 2) # 24*12
            self.corr3a_conv3 = ResBlock(256, 512, 2) # 12*6
            self.corr4a_conv1 = ResBlock(41, 64, 2) # 24*12
            self.corr4a_conv2 = ResBlock(64, 128, 2) # 12*6
        else:
            self.corr3a_conv1 = conv(self.batchNorm, 41, 128, 3, stride=2) # 48*24
            self.corr3a_conv2 = conv(self.batchNorm, 128, 256, 3, stride=2) # 24*12
            self.corr3a_conv3 = conv(self.batchNorm, 256, 512, 3, stride=2) # 12*6
            self.corr4a_conv1 = conv(self.batchNorm, 41, 64, 3, stride=2) # 24*12
            self.corr4a_conv2 = conv(self.batchNorm, 64, 128, 3, stride=2) # 12*6

        self.iconv1 = nn.ConvTranspose2d(512+128, 512, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(256+256+64+1, 384, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(128+128+41+1, 256, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(64+256+1, 128, 3, 1, 1)
        self.iconv5 = nn.ConvTranspose2d(128+128+1, 64, 3, 1, 1)
        self.iconv6 = nn.ConvTranspose2d(64+64+1, 32, 3, 1, 1)
        self.iconv7 = nn.ConvTranspose2d(32+3+1, 16, 3, 1, 1)

        self.pred1 = predict_flow(512) # 12*6
        self.upflow1to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upconv1 = deconv(512, 256) # 24*12
        self.pred2 = predict_flow(384)
        self.upflow2to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upconv2 = deconv(384, 128) # 48*24
        self.pred3 = predict_flow(256) # 
        self.upflow3to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upconv3 = deconv(256, 64) # 96*48
        self.pred4 = predict_flow(128) 
        self.upflow4to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upconv4 = deconv(128, 128)  # 192*96
        self.pred5 = predict_flow(64)
        self.upflow5to6 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upconv5 = deconv(64, 64) #384*192
        self.pred6 = predict_flow(32)
        self.upflow6to7 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upconv6 = deconv(32, 32) #768*384
        self.pred7 = predict_flow(16)

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
        conv3_l = self.conv3(conv2_l)
        conv3a_l = self.conv3a(conv3_l)

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)
        conv3a_r = self.conv3a(conv3_r)

        corr3a = self.corr3a(conv3a_l, conv3a_r)
        corr3a_act = self.corr3a_act(corr3a)
        corr3a_conv1 = self.corr3a_conv1(corr3a_act)
        corr3a_conv2 = self.corr3a_conv2(corr3a_conv1)
        corr3a_conv3= self.corr3a_conv3(corr3a_conv2)

        conv4_l = self.conv4(conv3a_l)
        conv4a_l = self.conv4a(conv4_l)
        conv4_r = self.conv4(conv3a_r)
        conv4a_r = self.conv4a(conv4_r)
        corr4a = self.corr4a(conv4a_l, conv4a_r)
        corr4a_act = self.corr4a_act(corr4a)
        corr4a_conv1 = self.corr4a_conv1(corr4a_act)
        corr4a_conv2 = self.corr4a_conv2(corr4a_conv1)

        concat1 = torch.cat((corr3a_conv3, corr4a_conv2),1)
        iconv1 = self.iconv1(concat1)
        pred1 = self.pred1(iconv1)
        upconv1 = self.upconv1(iconv1)
        upflow1to2 = self.upflow1to2(pred1) 

        concat2 = torch.cat((upflow1to2, corr3a_conv2, corr4a_conv1, upconv1), 1)
        iconv2 = self.iconv2(concat2)
        pred2 = self.pred2(iconv2)
        upconv2 = self.upconv2(iconv2)
        upflow2to3 = self.upflow2to3(pred2) 

        concat3 = torch.cat((upflow2to3, corr4a_act, corr3a_conv1, upconv2), 1)
        iconv3 = self.iconv3(concat3)
        pred3 = self.pred3(iconv3)
        upconv3 = self.upconv3(iconv3)
        upflow3to4 = self.upflow3to4(pred3) 

        concat4 = torch.cat((upflow3to4, conv3_l, upconv3), 1)
        iconv4 = self.iconv4(concat4)
        pred4 = self.pred4(iconv4)
        upconv4 = self.upconv4(iconv4)
        upflow4to5 = self.upflow4to5(pred4) 

        concat5 = torch.cat((upflow4to5, conv2_l, upconv4), 1)
        iconv5 = self.iconv5(concat5)
        pred5 = self.pred5(iconv5)
        upconv5 = self.upconv5(iconv5)
        upflow5to6 = self.upflow5to6(pred5) 

        concat6 = torch.cat((upflow5to6, conv1_l, upconv5), 1)
        iconv6 = self.iconv6(concat6)
        pred6 = self.pred6(iconv6)
        upconv7 = self.upconv6(iconv6)
        upflow6to7 = self.upflow6to7(pred6) 

        concat7 = torch.cat((upflow6to7, img_left, upconv7), 1)
        iconv7 = self.iconv7(concat7)
        pred7 = self.pred7(iconv7)
        return [pred7, pred6, pred5, pred4, pred3, pred2, pred1]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

