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
from networks.MobileDispNetC3 import MobileExtractNet, MobileDispCUNet
from networks.MobileDispNetRes3 import MobileDispNetRes
from networks.submodules import *
import copy
#from torch2trt import torch2trt

class MobileFADNet(nn.Module):

    def __init__(self, batchNorm=True, lastRelu=False, resBlock=True, maxdisp=-1, input_channel=3, input_img_shape=None, warp_size=None):
        super(MobileFADNet, self).__init__()
        self.input_channel = input_channel
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu
        self.maxdisp = maxdisp
        self.resBlock = resBlock

        # First Block (Extract)
        self.extract_network = MobileExtractNet(self.resBlock)

        # Second Block (MobileDispNetCUNet)
        self.dispcunet = MobileDispCUNet()

        # Third Block (DispNetRes), input is 11 channels(img0, img1, img1->img0, flow, diff-mag)
        in_planes = 3 * 3 + 1 + 1
        self.dispnetres = MobileDispNetRes(in_planes, self.batchNorm, lastRelu=self.lastRelu, maxdisp=self.maxdisp, input_channel=input_channel)

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
        self.model_trt = None

    def trt_transform(self):
        net = copy.deepcopy(self)
        x = torch.rand((1, 6, 576, 960)).cuda()
        net.extract_network = torch2trt(net.extract_network, [x])
    
        # extract features
        conv1_l, conv2_l, conv3a_l, conv3a_r = net.extract_network(x)
    
        # build corr
        out_corr = build_corr(conv3a_l, conv3a_r, max_disp=40)
    
        # generate first-stage flows
        net.dispcunet = torch2trt(net.dispcunet, [x, conv1_l, conv2_l, conv3a_l, out_corr])
        dispnetc_flows = net.dispcunet(x, conv1_l, conv2_l, conv3a_l, out_corr)
        dispnetc_final_flow = dispnetc_flows[0]
    
        diff_img0 = x[:, :3, :, :] - x[:, 3:, :, :]
        norm_diff_img0 = channel_length(diff_img0)
    
        # concat img0, img1, img1->img0, flow, diff-mag
        inputs_net2 = torch.cat((x, x[:, 3:, :, :], dispnetc_final_flow, norm_diff_img0), dim = 1)
    
        net.dispnetres = torch2trt(net.dispnetres, [inputs_net2, dispnetc_final_flow])

        return net


    def get_tensorrt_model(self):

        if self.model_trt == None:
            self.model_trt = self.trt_transform()
        return self.model_trt

    def forward(self, inputs):

        # split left image and right image
        # inputs = inputs_target[0]
        # target = inputs_target[1]
        imgs = torch.chunk(inputs, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        # extract features
        conv1_l, conv2_l, conv3a_l, conv3a_r = self.extract_network(inputs)

        # build corr
        out_corr = build_corr(conv3a_l, conv3a_r, max_disp=40)
        # generate first-stage flows
        dispnetc_flows = self.dispcunet(inputs, conv1_l, conv2_l, conv3a_l, out_corr)
        dispnetc_final_flow = dispnetc_flows[0]

        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        #dummy_flow = torch.autograd.Variable(torch.zeros(dispnetc_final_flow.data.shape).cuda())
        #dispnetc_final_flow_2d = torch.cat((dispnetc_final_flow, dummy_flow), dim = 1)
        #resampled_img1 = self.resample1(inputs[:, self.input_channel:, :, :], -dispnetc_final_flow_2d)
        #norm_diff_img0 = self.channelnorm(diff_img0)
        resampled_img1 = warp_right_to_left(inputs[:, self.input_channel:, :, :], -dispnetc_final_flow)
        diff_img0 = inputs[:, :self.input_channel, :, :] - resampled_img1
        norm_diff_img0 = channel_length(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        inputs_net2 = torch.cat((inputs, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim = 1)

        # dispnetres
        dispnetres_flows = self.dispnetres(inputs_net2, dispnetc_final_flow)
        index = 0
        #print('Index: ', index)
        dispnetres_final_flow = dispnetres_flows[index]
        

        if self.training:
            return dispnetc_flows, dispnetres_flows
        else:
            return dispnetc_final_flow, dispnetres_final_flow# , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1


    def weight_parameters(self):
    	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
    	return [param for name, param in self.named_parameters() if 'bias' in name]


