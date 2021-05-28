from __future__ import print_function
import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import skimage
import torch.cuda as ct
from net_builder import SUPPORT_NETS, build_net
import torch.nn.functional as F
import torch.nn as nn
from utils.common import count_parameters 
import psutil
from pytorch_utils import get_net_info
#from torch2trt import torch2trt

process = psutil.Process(os.getpid())
cudnn.benchmark = True

FP16=False

if FP16:
    import apex
else:
    apex = None


def load_model_trained_with_DP(net, state_dict):
    own_state = net.state_dict()
    for name, param in state_dict.items():
        own_state[name[7:]].copy_(param)

def check_tensorrt(y, y_trt):
    print(torch.max(torch.abs(y - y_trt)))

def detect(opt):

    net_name = opt.net
    enabled_tensorrt = opt.trt

    devices = [int(item) for item in opt.devices.split(',')]
    ngpu = len(devices)

    # build net according to the net name
    if net_name == "psmnet" or net_name == "ganet":
        net = build_net(net_name)(192)
    elif net_name in ['fadnet', 'mobilefadnet', 'slightfadnet', 'tinyfadnet', 'xfadnet']:
        eratio = 8; dratio = 8
        if net_name == 'mobilefadnet':
            eratio = 4; dratio = 4
        elif net_name == 'slightfadnet':
            eratio = 2; dratio = 2
        elif net_name == 'tinyfadnet':
            eratio = 1; dratio = 1
        elif net_name == 'xfadnet':
            eratio = 16; dratio = 16
        net = build_net(net_name)(maxdisp=192, encoder_ratio=eratio, decoder_ratio=dratio)

    #elif net_name == "mobilefadnet":
    #    #B, max_disp, H, W = (wopt.batchSize, 40, 72, 120)
    #    shape = (opt.batchSize, 40, 72, 120) #TODO: Should consider how to dynamically use
    #    warp_size = (opt.batchSize, 3, 576, 960)
    #    net = build_net(net_name)(batchNorm=False, lastRelu=True, input_img_shape=shape, warp_size=warp_size)

    if ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=devices)

    num_of_parameters = count_parameters(net)
    print('Model: %s, # of parameters: %d' % (net_name, num_of_parameters))

    net = net.cuda()
    get_net_info(net, input_shape=(3, 576, 960))
    if enabled_tensorrt:
        net = net.get_tensorrt_model()
    #torch.save(net.state_dict(), 'models/mobilefadnet_trt.pth')
    if FP16:
        net = apex.amp.initialize(net, None, opt_level='O2') 
    net.eval()

    # fake input data
    dummy_input = torch.randn(1, 6, 576, 960, dtype=torch.float).cuda()

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 30
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    with torch.no_grad():
        for _ in range(10):
            _ = net(dummy_input, enabled_tensorrt)
            # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = net(dummy_input, enabled_tensorrt)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            print(rep, curr_time)
    
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='fadnet', choices=SUPPORT_NETS)
    parser.add_argument('--devices', type=str, help='devices', default='0')
    parser.add_argument('--batchSize', type=int, help='mini batch size', default=1)
    parser.add_argument('--trt', action='store_true', help='enables tensorrt')

    opt = parser.parse_args()
    detect(opt)
