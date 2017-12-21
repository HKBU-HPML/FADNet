#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:So 04 Jun 2017 22:39:19 CEST
Info: flownet model
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nets.model import Model

class FlowNetC(Model):

    def __init__(self, args):
        super(FlowNetC,self).__init__(args)
        self.conv1a = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.conv1b = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.conv2a = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv2b = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3a = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3b = nn.Conv2d(64, 128, 5, stride=2, padding=2)

    def _init_weights(self):
        pass

    def _leaky_relu(self, input_):
        return F.leaky_relu(input_, negative_slope=0.1, inplace=True)

    def forward(self, imga, imgb):
        conv1_imga = self._leaky_relu(self.conv1a(imga))
        conv1_imgb = self._leaky_relu(self.conv1b(imgb))
        conv2_imga = self._leaky_relu(self.conv2a(conv1_imga))
        conv2_imgb = self._leaky_relu(self.conv2b(conv1_imgb)) 
        conv3_imga = self._leaky_relu(self.conv3a(conv2_imga)) 
        conv3_imgb = self._leaky_relu(self.conv3b(conv2_imgb)) 
        # correlation layer

class CorrelationLayer(Model):

    def __init__(self, args=None, padding=20, kernel_size = 1, max_displacement=20, stride_1=1, stride_2=2):
        # TODO generilize kernel size (right now just 1)
        # TODO generilize stride_1 (right now just 1), cause there is no downsample layer in pytorch
        super(CorrelationLayer,self).__init__(args)
        self.pad = padding
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride_1 = stride_1
        self.stride_2 = stride_2


    def forward(self, x_1, x_2):
        """
        Arguments
        ---------
        x_1 : 4D torch.Tensor (bathch channel height width)
        x_2 : 4D torch.Tensor (bathch channel height width)

        """
        x_1 = x_1.transpose(1,2).transpose(2,3)
        x_2 = F.pad(x_2, tuple([self.pad for _ in range(4)])).transpose(1,2).transpose(2,3)
        mean_x_1 = torch.mean(x_1,3) 
        mean_x_2 = torch.mean(x_2,3) 
        sub_x_1 = x_1.sub(mean_x_1.expand_as(x_1))
        sub_x_2 = x_2.sub(mean_x_2.expand_as(x_2))
        st_dev_x_1 = torch.std(x_1,3) 
        st_dev_x_2 = torch.std(x_2,3)
        
        # TODO need optimize
        out_vb = torch.zeros(1)
        _y=0
        _x=0
        while _y < self.max_displacement*2+1:
            while _x < self.max_displacement*2+1:
                c_out = (torch.sum(sub_x_1*sub_x_2[:,_x:_x+x_1.size(1),
                    _y:_y+x_1.size(2),:],3) / 
                (st_dev_x_1*st_dev_x_2[:,_x:_x+x_1.size(1),
                    _y:_y+x_1.size(2),:])).transpose(2,3).transpose(1,2)
                out_vb = torch.cat((out_vb,c_out),1) if len(out_vb.size())!=1 else c_out
                _x += self.stride_2
            _y += self.stride_2
        return out_vb 
