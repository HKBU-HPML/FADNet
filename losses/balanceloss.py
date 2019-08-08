from torch.autograd import Function, Variable
import torch
from torch.nn.modules.module import Module

class MyLoss2Function(Function):
    def __init__(self, thresh=1, alpha=2):
        self.thresh = thresh
        self.alpha = alpha
    def forward(self, input1, input2):
        self.diff = input1 - input2
        temp=torch.abs(self.diff)
        temp[temp < self.thresh] = temp[temp < self.thresh] ** 2 / self.thresh
        tag = (temp <= self.thresh + self.alpha) & (temp >= self.thresh)
        temp[tag]=temp[tag] * 2 - (temp[tag] - self.thresh) ** 2 /(2.0 * self.alpha) - self.thresh
        temp[temp > self.thresh + self.alpha] += (self.alpha / 2.0)
        
        return torch.mean(temp)
    def backward(self, gradOutput):
        scale = torch.abs(self.diff)
        scale[scale > self.thresh + self.alpha] = 1
        tag = (scale <= self.thresh+self.alpha) & (scale >= self.thresh)
        scale[tag] = 2 - (scale[tag] - self.thresh) / self.alpha
        tag = scale < self.thresh
        scale[tag] = 2*scale[tag] / self.thresh
        self.diff[self.diff > 0] = 1.0
        self.diff[self.diff < 0] = -1.0
        self.diff = self.diff * scale * gradOutput / scale.numel()
        return self.diff, Variable(torch.Tensor([0]))


class MyLoss2(Module):
    def __init__(self, thresh=1, alpha=2):
        super(MyLoss2, self).__init__()
        self.thresh = thresh
        self.alpha = alpha
    def forward(self, input1, input2):
        result = MyLoss2Function(self.thresh, self.alpha)(input1, input2)
        return result
