from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from networks.DispNetC import ExtractNet, CUNet
from networks.DispNetS import DispNetS
from networks.submodules import *

class DispNetCSS(nn.Module):

    def __init__(self, resBlock=True, maxdisp=192, input_channel=3, encoder_ratio=16, decoder_ratio=16):
        super(DispNetCSS, self).__init__()
        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.resBlock = resBlock
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio

        # First Block (Extract)
        self.extract_network = ExtractNet(resBlock=resBlock, maxdisp=self.maxdisp, input_channel=input_channel, encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)

        # Second Block (CUNet)
        self.cunet = CUNet(resBlock=resBlock, maxdisp=self.maxdisp, input_channel=input_channel, encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)

        # Third Block (DispNetRes), input is 11 channels(img0, img1, img1->img0, flow, diff-mag)
        in_planes = 3 * 3 + 1 + 1
        self.dispnets1 = DispNetS(in_planes, resBlock=self.resBlock, input_channel=3, encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)
        self.dispnets2 = DispNetS(in_planes, resBlock=self.resBlock, input_channel=3, encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)

        self.relu = nn.ReLU(inplace=False)

        self.model_trt = None

    def forward(self, inputs, enabled_tensorrt=False):

        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        # extract features
        conv1_l, conv2_l, conv3a_l, conv3a_r = self.extract_network(inputs)

        # build corr
        out_corr = build_corr(conv3a_l, conv3a_r, max_disp=self.maxdisp//8+16)
        # generate first-stage flows
        dispnetc_flows = self.cunet(inputs, conv1_l, conv2_l, conv3a_l, out_corr)
        dispnetc_final_flow = dispnetc_flows[0]

        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1 = warp_right_to_left(inputs[:, self.input_channel:, :, :], -dispnetc_final_flow)
        diff_img0 = inputs[:, :self.input_channel, :, :] - resampled_img1
        norm_diff_img0 = channel_length(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-img
        inputs_net2 = torch.cat((inputs, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim = 1)

        # dispnets1
        dispnets1_flows = self.dispnets1(inputs_net2)
        dispnets1_final_flow = dispnets1_flows[0]

        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1s = warp_right_to_left(inputs[:, self.input_channel:, :, :], -dispnets1_final_flow)
        diff_img0s = inputs[:, :self.input_channel, :, :] - resampled_img1s
        norm_diff_img0s = channel_length(diff_img0s)

        # concat img0, img1, img1->img0, flow, diff-img
        inputs_net3 = torch.cat((inputs, resampled_img1s, dispnets1_final_flow, norm_diff_img0s), dim = 1)

        # dispnets2
        dispnets2_flows = self.dispnets2(inputs_net3)
        dispnets2_final_flow = dispnets2_flows[0]

        if self.training:
            return dispnetc_flows, dispnets1_flows, dispnets2_flows
        else:
            return dispnetc_final_flow, dispnets1_final_flow, dispnets2_final_flow # , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


