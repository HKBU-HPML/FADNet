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

def load_model_trained_with_DP(net, state_dict):
    own_state = net.state_dict()
    for name, param in state_dict.items():
        own_state[name[7:]].copy_(param)

def check_tensorrt(y, y_trt):
    print(torch.max(torch.abs(y - y_trt)))

def detect(opt):

    net_name = opt.net
    model = opt.model
    result_path = opt.rp
    file_list = opt.filelist
    filepath = opt.filepath

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    devices = [int(item) for item in opt.devices.split(',')]
    ngpu = len(devices)

    # build net according to the net name
    if net_name == "psmnet" or net_name == "ganet":
        net = build_net(net_name)(192)
    elif net_name in ['fadnet', 'mobilefadnet', 'slightfadnet', 'xfadnet']:
        eratio = 4; dratio = 4
        if net_name == 'mobilefadnet':
            eratio = 2; dratio = 2
        elif net_name == 'slightfadnet':
            eratio = 1; dratio = 1
        elif net_name == 'xfadnet':
            eratio = 8; dratio = 8
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

    net.eval()
    net = net.cuda()
    get_net_info(net, input_shape=(3, 576, 960))
    #net = net.get_tensorrt_model()
    #torch.save(net.state_dict(), 'models/mobilefadnet_trt.pth')

    # fake input data
    dummy_input = torch.randn(1, 6, 576, 960, dtype=torch.float).cuda()

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 30
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    with torch.no_grad():
        for _ in range(10):
            _ = net(dummy_input)
            # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = net(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            print(rep, curr_time)
    
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)


    #s = time.time()

    #avg_time = []
    #display = 50
    #warmup = 10
    #for i, sample_batched in enumerate(test_loader):
    #    #if i > 215:
    #    #    break
    #    stime = time.time()

    #    input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)

    #    print('input Shape: {}'.format(input.size()))
    #    num_of_samples = input.size(0)

    #    #output, input_var = detect_batch(net, sample_batched, opt.net, (540, 960))

    #    input = input.cuda()
    #    input_var = input #torch.autograd.Variable(input, volatile=True)
    #    input_var = F.interpolate(input_var, (576, 960), mode='bilinear')
    #    iotime = time.time()
    #    print('[{}] IO time:{}'.format(i, iotime-stime))

    #    if i > warmup:
    #        ss = time.time()

    #    with torch.no_grad():
    #        if opt.net == "psmnet" or opt.net == "ganet":
    #            output = net(input_var)
    #            output = output.unsqueeze(1)
    #        elif opt.net == "dispnetc":
    #            output = net(input_var)[0]
    #        else:
    #            output = net(input_var)[-1]
    #    itime = time.time()
    #    print('[{}] Inference time:{}'.format(i, itime-iotime))
 
    #    if i > warmup:
    #        avg_time.append((time.time() - ss))
    #        if (i - warmup) % display == 0:
    #            print('Average inference time: %f' % np.mean(avg_time))
    #            mbytes = 1024.*1024
    #            print('GPU memory usage memory_allocated: %d MBytes, max_memory_allocated: %d MBytes, memory_cached: %d MBytes, max_memory_cached: %d MBytes, CPU memory usage: %d MBytes' %  \
    #                (ct.memory_allocated()/mbytes, ct.max_memory_allocated()/mbytes, ct.memory_cached()/mbytes, ct.max_memory_cached()/mbytes, process.memory_info().rss/mbytes))
    #            avg_time = []

    #    print('[%d] output shape:' % i, output.size())
    #    output = scale_disp(output, (output.size()[0], 540, 960))
    #    disp = output[:, 0, :, :]
    #    ptime = time.time()
    #    print('[{}] Post-processing time:{}'.format(i, ptime-itime))

    #    for j in range(num_of_samples):

    #        name_items = sample_batched['img_names'][0][j].split('/')
    #        # write disparity to file
    #        output_disp = disp[j]
    #        np_disp = disp[j].float().cpu().numpy()

    #        print('Batch[{}]: {}, average disp: {}({}-{}).'.format(i, j, np.mean(np_disp), np.min(np_disp), np.max(np_disp)))
    #        save_name = '_'.join(name_items).replace(".png", "_d.png")# for girl02 dataset
    #        print('Name: {}'.format(save_name))
    #        skimage.io.imsave(os.path.join(result_path, save_name),(np_disp*256).astype('uint16'))
    #    print('Current batch time used:: {}'.format(time.time()-stime))


    #        #save_name = '_'.join(name_items).replace("png", "pfm")# for girl02 dataset
    #        #print('Name: {}'.format(save_name))
    #        #np_disp = np.flip(np_disp, axis=0)
    #        #save_pfm('{}/{}'.format(result_path, save_name), np_disp)
    #        


    #print('Evaluation time used: {}'.format(time.time()-s))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='fadnet', choices=SUPPORT_NETS)
    parser.add_argument('--model', type=str, help='model to load', default='best.pth')
    parser.add_argument('--filelist', type=str, help='file list', default='FlyingThings3D_release_TEST.list')
    parser.add_argument('--filepath', type=str, help='file path', default='./data')
    parser.add_argument('--devices', type=str, help='devices', default='0')
    parser.add_argument('--display', type=int, help='Num of samples to print', default=10)
    parser.add_argument('--rp', type=str, help='result path', default='./result')
    parser.add_argument('--flowDiv', type=float, help='flow division', default='1.0')
    parser.add_argument('--batchSize', type=int, help='mini batch size', default=1)

    opt = parser.parse_args()
    detect(opt)
