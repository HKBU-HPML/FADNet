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
from pytorch_memlab import MemReporter

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
    print('network_name: ', net_name)
    if net_name in ["gwcnet", "ganet", "psmnet"]:
        net = build_net(net_name)(192)
    elif net_name in ['fadnet', 'mobilefadnet', 'slightfadnet', 'tinyfadnet', 'microfadnet', 'xfadnet']:
        eratio = 16; dratio = 16
        if net_name == 'mobilefadnet':
            eratio = 8; dratio = 8
        elif net_name == 'slightfadnet':
            eratio = 4; dratio = 4
        elif net_name == 'tinyfadnet':
            eratio = 2; dratio = 1
        elif net_name == 'microfadnet':
            eratio = 1; dratio = 1
        elif net_name == 'xfadnet':
            eratio = 32; dratio = 32
        net = build_net(net_name)(maxdisp=192, encoder_ratio=eratio, decoder_ratio=dratio)
    elif net_name in ["dispnetc", "dispnetcss"]:
        net = build_net(net_name)(resBlock=False, maxdisp=192)
    elif net_name == "crl":
        net = build_net('fadnet')(resBlock=False, maxdisp=192)
    #elif net_name == "mobilefadnet":
    #    #B, max_disp, H, W = (wopt.batchSize, 40, 72, 120)
    #    shape = (opt.batchSize, 40, 72, 120) #TODO: Should consider how to dynamically use
    #    warp_size = (opt.batchSize, 3, 576, 960)
    #    net = build_net(net_name)(batchNorm=False, lastRelu=True, input_img_shape=shape, warp_size=warp_size)

    if ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=devices)

    num_of_parameters = count_parameters(net)
    print('Model: %s, # of parameters: %d' % (net_name, num_of_parameters))

    if FP16:
        net = apex.amp.initialize(net, None, opt_level='O2') 
    net = net.cuda()
    reporter = MemReporter(net)

    width = 960
    height = 576
    get_net_info(net, input_shape=(3, height, width))
    if enabled_tensorrt:
        net = net.get_tensorrt_model((1, 6, height, width))
    #torch.save(net.state_dict(), 'models/mobilefadnet_trt.pth')
    # fake input data
    dummy_input = torch.randn(1, 6, height, width, dtype=torch.float).cuda()

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 30
    mbytes = 2**20
    timings=np.zeros((repetitions,1))
    net.eval()
    #GPU-WARM-UP
    with torch.no_grad():
        for _ in range(10):
            _ = net(dummy_input, enabled_tensorrt)
            # MEASURE PERFORMANCE
    torch.cuda.empty_cache()
    time.sleep(1)
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = net(dummy_input, enabled_tensorrt)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            reporter.report()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            print(rep, curr_time)
    
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(net_name, mean_syn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='fadnet', choices=SUPPORT_NETS)
    parser.add_argument('--devices', type=str, help='devices', default='0')
    parser.add_argument('--batchSize', type=int, help='mini batch size', default=1)
    parser.add_argument('--trt', action='store_true', help='enables tensorrt')

    opt = parser.parse_args()
    detect(opt)
