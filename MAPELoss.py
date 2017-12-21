import torch
import torch.nn as nn
import math
import numpy as np 

class MAPELoss(nn.Module):

    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, input, target):

        loss_layer = nn.L1Loss()
        errMap = input - target
        errMap = errMap / (target + 1)
        mape = loss_layer(errMap, errMap.detach()*0)

        return mape

