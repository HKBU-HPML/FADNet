import torch
import horovod.torch as hvd

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HVDMetric(object):
    def __init__(self, name):
        self.name = name
        self.sum = 0 #torch.tensor(0.)
        self.n = 0 #torch.tensor(0.)
        self.val = 0.

    def update(self, val, n=1):
        self.val = val.item()
        self.sum += float(hvd.allreduce(val.detach().cpu() * n, name=self.name, average=False))
        self.n += n * hvd.size()

    @property
    def avg(self):
        return self.sum / self.n

