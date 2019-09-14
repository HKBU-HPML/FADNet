from __future__ import print_function
import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import skimage
#from dispnet import *
#from networks.dispnet_v2 import *
import torch.cuda as ct
from networks.DispNetCSRes import DispNetCSRes
from net_builder import SUPPORT_NETS, build_net
from losses.multiscaleloss import multiscaleloss
import torch.nn.functional as F
#from dataset import DispDataset, save_pfm, RandomRescale
from dataloader.SceneFlowLoader import DispDataset
from utils.preprocess import scale_disp, save_pfm
from utils.common import count_parameters 
from torch.utils.data import DataLoader
from torchvision import transforms
import psutil

process = psutil.Process(os.getpid())
cudnn.benchmark = True

#input_transform = transforms.Compose([
#        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
#        # transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
#        ])
#
#target_transform = transforms.Compose([
#        transforms.Normalize(mean=[0],std=[1.0])
#        ])

def detect(opt):
    model = opt.model
    result_path = opt.rp
    file_list = opt.filelist
    filepath = opt.filepath
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    devices = [int(item) for item in opt.devices.split(',')]
    ngpu = len(devices)
    #net = DispNetC(ngpu, True)
    #net = DispNetCSRes(ngpu, False, True)
    #net = DispNetCSResWithMono(ngpu, False, True, input_channel=3)
    if opt.net == "psmnet" or opt.net == "ganet":
        net = build_net(opt.net)(192)
    elif opt.net == "dispnetc":
        net = build_net(opt.net)(batchNorm=False, lastRelu=True, resBlock=False)
    else:
        net = build_net(opt.net)(batchNorm=False, lastRelu=True)
    #net = build_net(opt.net)(ngpu, False, True)
 
    #model_data = torch.load(model)
    #print(model_data.keys())
    #if 'state_dict' in model_data.keys():
    #    net.load_state_dict(model_data['state_dict'])
    #else:
    #    net.load_state_dict(model_data)

    num_of_parameters = count_parameters(net)
    print('Model: %s, # of parameters: %d' % (opt.net, num_of_parameters))

    net = torch.nn.DataParallel(net, device_ids=devices).cuda()
    net.eval()

    batch_size = int(opt.batchSize)
    test_dataset = DispDataset(txt_file=file_list, root_dir=filepath, phase='detect')
    test_loader = DataLoader(test_dataset, batch_size = batch_size, \
                        shuffle = False, num_workers = 1, \
                        pin_memory = True)

    s = time.time()
    #high_res_EPE = multiscaleloss(scales=1, downscale=1, weights=(1), loss='L1', sparse=False)

    avg_time = []
    display = 100
    warmup = 10
    for i, sample_batched in enumerate(test_loader):
        input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)
        # print('input Shape: {}'.format(input.size()))
        num_of_samples = input.size(0)
        target = sample_batched['gt_disp']

        #print('disp Shape: {}'.format(target.size()))
        #original_size = (1, target.size()[2], target.size()[3])

        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        
        if i > warmup:
            ss = time.time()
        output = net(input_var)[1] 
        if i > warmup:
            avg_time.append((time.time() - ss))
            if (i - warmup) % display == 0:
                print('Average inference time: %f' % np.mean(avg_time))
                mbytes = 1024.*1024
                print('GPU memory usage memory_allocated: %d MBytes, max_memory_allocated: %d MBytes, memory_cached: %d MBytes, max_memory_cached: %d MBytes, CPU memory usage: %d MBytes' %  \
                    (ct.memory_allocated()/mbytes, ct.max_memory_allocated()/mbytes, ct.memory_cached()/mbytes, ct.max_memory_cached()/mbytes, process.memory_info().rss/mbytes))
                avg_time = []
        #np_depth = scale_disp(output)
        #for j in range(num_of_samples):
        #    # scale back depth
        #    np_depth = output[j] #.data.cpu().numpy()
        #    #print(np.min(np_depth), np.max(np_depth))
        #    #cuda_depth = torch.from_numpy(np_depth).cuda()
        #    #cuda_depth = torch.autograd.Variable(cuda_depth, volatile=True)

        #    # flow2_EPE = high_res_EPE(output[j], target_var[j]) * 1.0
        #    #flow2_EPE = high_res_EPE(cuda_depth, target_var[j]) * 1.0
        #    #print('Shape: {}'.format(output[j].size()))
        #    print('Batch[{}]: {}, average disp: {}'.format(i, j, np.mean(np_depth)))
        #    #print('Batch[{}]: {}, Flow2_EPE: {}'.format(i, sample_batched['img_names'][0][j], flow2_EPE.data.cpu().numpy()))

        #    name_items = sample_batched['img_names'][0][j].split('/')
        #    #save_name = '_'.join(name_items).replace('.png', '.pfm')# for girl02 dataset
        #    save_name = '_'.join(name_items)# for girl02 dataset
        #    #save_name = 'predict_{}_{}_{}.pfm'.format(name_items[-4], name_items[-3], name_items[-1].split('.')[0])
        #    #save_name = 'predict_{}_{}.pfm'.format(name_items[-1].split('.')[0], name_items[-1].split('.')[1])
        #    #save_name = 'predict_{}.pfm'.format(name_items[-1])
        #    img = np.flip(np_depth[0], axis=0)
        #    print('Name: {}'.format(save_name))
        #    print('')
        #    #save_pfm('{}/{}'.format(result_path, save_name), img)
        #    skimage.io.imsave(os.path.join(result_path, save_name),(img*256).astype('uint16'))
            


    print('Evaluation time used: {}'.format(time.time()-s))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='dispnetcres', choices=SUPPORT_NETS)
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
