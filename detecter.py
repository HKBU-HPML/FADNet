from __future__ import print_function
import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
from dispnet import *
from multiscaleloss import multiscaleloss
from dataset import DispDataset, save_pfm, RandomRescale
from torch.utils.data import DataLoader
from torchvision import transforms

input_transform = transforms.Compose([
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        # transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
        ])

target_transform = transforms.Compose([
        transforms.Normalize(mean=[0],std=[1.0])
        ])


def detect(model, result_path, file_list, filepath):
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    devices = [int(item) for item in opt.devices.split(',')]
    ngpu = len(devices)
    net = DispNetCSRes(ngpu, False)

    model_data = torch.load(model)
    print(model_data.keys())
    if 'state_dict' in model_data.keys():
        net.load_state_dict(model_data['state_dict'])
    else:
        net.load_state_dict(model_data)
    net = torch.nn.DataParallel(net, device_ids=devices).cuda()
    net.eval()

    batch_size = 4
    
    test_dataset = DispDataset(txt_file=file_list, root_dir=filepath, transform=[input_transform, target_transform], phase='test')
    test_loader = DataLoader(test_dataset, batch_size = batch_size, \
                        shuffle = False, num_workers = 1, \
                        pin_memory = True)
    s = time.time()

    high_res_EPE = multiscaleloss(scales=1, downscale=1, weights=(1), loss='L1', sparse=False)

    for i, sample_batched in enumerate(test_loader):
        input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)
        print('input Shape: {}'.format(input.size()))
        num_of_samples = input.size(0)
        target = sample_batched['gt_disp']
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        output = net(input_var)

        for j in range(num_of_samples):
            flow2_EPE = high_res_EPE(output[j], target_var[j]) * 1.0
            #print('Shape: {}'.format(output[j].size()))
            print('Batch[{}]: {}, Flow2_EPE: {}'.format(i, j, flow2_EPE.data.cpu().numpy()))

            name_items = sample_batched['img_names'][0][j].split('/')
            save_name = 'predict_{}_{}_{}.pfm'.format(name_items[-4], name_items[-3], name_items[-1].split('.')[0])
            img = output[j].data.cpu().numpy()
            img = RandomRescale.scale_back(img, orignal_size=(1, 540, 960))
            img = np.flip(img[0], axis=0)
            print('Name: {}'.format(save_name))
            save_pfm('{}/{}'.format(result_path, save_name), img)

    print('Evaluation time used: {}'.format(time.time()-s))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model to load', default='best.pth')
    parser.add_argument('--filelist', type=str, help='file list', default='FlyingThings3D_release_TEST.list')
    parser.add_argument('--filepath', type=str, help='file path', default='./data')
    parser.add_argument('--devices', type=str, help='devices', default='0')
    parser.add_argument('--display', type=int, help='Num of samples to print', default=10)
    parser.add_argument('--rp', type=str, help='result path', default='./result')
    opt = parser.parse_args()
    detect(opt.model, opt.rp, opt.filelist, opt.filepath)
