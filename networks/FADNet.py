from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from networks.DispNetC import ExtractNet, CUNet
from networks.DispNetRes import DispNetRes
from networks.submodules import *
import copy
from torch2trt import torch2trt

class FADNet(nn.Module):

    def __init__(self, resBlock=True, maxdisp=192, input_channel=3, encoder_ratio=8, decoder_ratio=8):
        super(FADNet, self).__init__()
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
        self.dispnetres = DispNetRes(in_planes, resBlock=resBlock, input_channel=input_channel, encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)

        self.relu = nn.ReLU(inplace=False)

        self.model_trt = None

    def trt_transform(self):
        net = copy.deepcopy(self)
        x = torch.rand((1, 6, 576, 960)).cuda()
        net.extract_network = torch2trt(net.extract_network, [x])
    
        # extract features
        conv1_l, conv2_l, conv3a_l, conv3a_r = net.extract_network(x)
    
        # build corr
        out_corr = build_corr(conv3a_l, conv3a_r, max_disp=self.maxdisp//8+16)
    
        # generate first-stage flows
        in_channels = out_corr.size(1) + conv3a_l.size(1)//4
        net.cunet.conv3_1.fix_dynamic(in_channels)
        net.dispcunet = torch2trt(net.cunet, [x, conv1_l, conv2_l, conv3a_l, out_corr])
        dispnetc_flows = net.dispcunet(x, conv1_l, conv2_l, conv3a_l, out_corr)
        dispnetc_final_flow = dispnetc_flows[0]
    
        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1 = warp_right_to_left(x[:, 3:, :, :], -dispnetc_final_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = channel_length(diff_img0)
    
        # concat img0, img1, img1->img0, flow, diff-mag
        inputs_net2 = torch.cat((x, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim = 1)
        #inputs_net2 = torch.cat((x, x[:, 3:, :, :], dispnetc_final_flow, norm_diff_img0), dim = 1)
    
        net.dispnetres = torch2trt(net.dispnetres, [inputs_net2, dispnetc_final_flow])

        return net


    def get_tensorrt_model(self):

        if self.model_trt == None:
            self.model_trt = self.trt_transform()
        return self.model_trt


    def forward(self, inputs, enabled_tensorrt = False):

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

        # dispnetres
        #dispnetres_flows = self.dispnetres(inputs_net2, dispnetc_flows)
        if enabled_tensorrt:
            dispnetres_flows = self.dispnetres(inputs_net2, dispnetc_final_flow)
        else:
            dispnetres_flows = self.dispnetres(inputs_net2, dispnetc_flows)

        index = 0
        dispnetres_final_flow = dispnetres_flows[index]
        

        if self.training:
            return dispnetc_flows, dispnetres_flows
        else:
            return dispnetc_final_flow, dispnetres_final_flow

    def weight_parameters(self):
    	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
    	return [param for name, param in self.named_parameters() if 'bias' in name]


