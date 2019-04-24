from __future__ import print_function

import torch
import torch.nn as nn

from torch.nn.init import kaiming_normal
from correlation_package.modules.corr import Correlation1d # from PWC-Net
from networks.submodules import ResBlock, conv, predict_flow, deconv

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1   = conv(True, 3, 64, 7, 2)
        self.conv2   = ResBlock(64, 96, 1)
        self.conv2_half   = conv(False, 96, 96, 3, 2)
        self.corr1 = Correlation1d(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=1, corr_multiply=1)
        self.corr1_conv1 = conv(False, 41, 96, 3, 2)

        self.corr2 = Correlation1d(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=1, corr_multiply=1)
        self.corr2_activation = nn.ReLU(inplace=True)
        self.corr2_conv1 = conv(False, 41, 96, 1, 1)

        self.downconv4 = conv(True, 192, 128)
        self.downconv5   = conv(True, 128, 256, stride=2)
        self.downconv6   = conv(True, 256, 256, stride=2)

        self.upconv6 = deconv(256, 256)
        self.upconv5 = deconv(256, 256)
        self.upconv4 = deconv(256, 128)
        self.upconv3 = deconv(128, 64)
        self.pred_res = predict_flow(64)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()

    def forward(self, input):
        imgs = torch.chunk(input, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        conv1_l = self.conv1(img_left)
        conv2_l = self.conv2(conv1_l)
        conv2_l_half = self.conv2_half(conv2_l)

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv2_r_half = self.conv2_half(conv2_r)

        out_corr1 = self.corr1(conv2_l, conv2_r)
        corr1_half = self.corr1_conv1(out_corr1)

        out_corr2 = self.corr2(conv2_l_half, conv2_r_half)
        out_corr2 = self.corr2_conv1(out_corr2)

        corr_cat = torch.cat((corr1_half, out_corr2), 1)
        x = self.downconv4(corr_cat)
        x = self.downconv5(x)
        x = self.downconv6(x)
        x = self.upconv6(x)
        x = self.upconv5(x)
        x = self.upconv4(x)
        x = self.upconv3(x)
        o = self.pred_res(x)
        return o

