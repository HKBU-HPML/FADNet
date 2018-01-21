import torch
import torch.nn as nn
import math
import numpy as np

def EPE(input_flow, target_flow):
    # print(input_flow.size(), target_flow.size())
    # print(target_flow - input_flow)
    EPE_map = torch.norm(target_flow - input_flow + 1e-16, 2, 1)
    # EPE_map = torch.abs(target_flow - input_flow + 1e-16)# / #, 2, 1)
    # print(target_flow.sum())
    # print(input_flow.sum())
    return EPE_map.mean()

class MultiScaleLoss(nn.Module):

    def __init__(self, scales, downscale, weights=None, loss='L1'):
        super(MultiScaleLoss, self).__init__()
        self.downscale = downscale
        self.weights = torch.Tensor(scales).fill_(1) if weights is None else torch.Tensor(weights)
        assert(len(self.weights) == scales)

        if type(loss) is str:

            if loss == 'L1':
                self.loss = nn.L1Loss()
            elif loss == 'MSE':
                self.loss = nn.MSELoss()
            elif loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
            elif loss == 'MAPE':
                self.loss = MAPELoss()
        else:
            self.loss = loss
        self.multiScales = [nn.AvgPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]
        # self.multiScales = [nn.functional.adaptive_avg_pool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]

    def forward(self, input, target):
        if type(input) is tuple:
            out = 0
            for i, input_ in enumerate(input):
                target_ = self.multiScales[i](target)
                EPE_ = EPE(input_, target_)
                out += self.weights[i] * EPE_
        else:
            out = self.loss(input, self.multiScales[0](target))
        return out

def multiscaleloss(scales=5, downscale=4, weights=None, loss='L1', sparse=False):
    if weights is None:
        weights = (0.005, 0.01, 0.02, 0.08, 0.32)
    if scales == 1 and type(weights) is not tuple:
        weights = (weights, )
    return MultiScaleLoss(scales, downscale, weights, loss)
