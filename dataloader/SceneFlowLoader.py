from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from utils.preprocess import *

class DispDataset(Dataset):

    def __init__(self, txt_file, root_dir, transform = None, phase='train', augment=False, center_crop=False):
        """
        Args:
            txt_file [string]: Path to the image list
            transform (callable, optional): Optional transform to be applied                on a sample
        """
        with open(txt_file, "r") as f:
            self.imgPairs = f.readlines()

        self.root_dir = root_dir
        #self.transform = transform
        #self.phase = phase
        #self.augment = augment 
        #self.center_crop = center_crop

    def __len__(self):
        return len(self.imgPairs)

    def __getitem__(self, idx):
        img_names = self.imgPairs[idx].rstrip().split()

        img_left_name = os.path.join(self.root_dir, img_names[0])
        img_right_name = os.path.join(self.root_dir, img_names[1])
        gt_disp_name = os.path.join(self.root_dir, img_names[2])

        def load_rgb(filename):

            if filename.find('.npy') > 0:
                filename = np.load(filename)
            else:
                filename = io.imread(filename)
           

        def load_disp(filename):
            gt_disp = None
            if gt_disp_name.endswith('pfm'):
                gt_disp, scale = load_pfm(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            elif gt_disp_name.endswith('npy'):
                gt_disp = np.load(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            else:
                gt_disp = Image.open(gt_disp_name)
                gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256

            return gt_disp

        img_left = load_rgb(img_left_name)
        img_right = load_rgb(img_right_name)
        gt_disp = load_disp(gt_disp_name)

        sample = {'img_left': img_left, 
                  'img_right': img_right, 
                  'gt_disp' : gt_disp,   
                 }

        if self.phase == 'test':
            #scale = RandomRescale((512, 512))
            #scale = RandomRescale((384, 768))
            #scale = RandomRescale((512, 512))
            scale = RandomRescale((1024, 1024))
            #scale = RandomRescale((1024+256, 1024+256)) # real
            #scale = RandomRescale((1024+128, 1024+128))
            #scale = RandomRescale((1024-128,1024-128))
            #scale = RandomRescale((768, 1280)) # Flying things
            #scale = RandomRescale((256, 768)) # KITTI
            #scale = RandomRescale((256, 512)) # KITTI
            #scale = RandomRescale((512, 512)) # girl
            #scale = RandomRescale((512 * 3, 896 * 3))
            #scale = RandomRescale((768-256, 1024-384)) 
            #scale = RandomRescale((768+128, 1024+128)) 
            #scale = RandomRescale((768, 1024 + 512))
            #scale = RandomRescale((1536, 1536)) # real data
            #scale = RandomRescale((2048, 3072)) # moto
            #scale = RandomRescale((512, 512))
            
            sample = scale(sample)
            pass

        tt = ToTensor()
        if self.transform:
            if gt_disp is None or img_left is None or img_right is None:
                print("sample data is none:", gt_disp_name)
                print('left: ', img_left_name)
                print('right: ', img_right_name)
                print('gt_disp: ', gt_disp_name)
                raise 
            sample['img_left'] = self.transform[0](tt(sample['img_left']))
            sample['img_right'] = self.transform[0](tt(sample['img_right']))
            sample['gt_disp'] = self.transform[1](tt(sample['gt_disp']))

        if self.phase != 'test':
            #crop = RandomCrop((384, 768))
            if self.center_crop == True:
                crop = CenterCrop((384, 768))
            else:
                crop = RandomCrop((384, 768)) # flyingthing, monkaa, driving
            #crop = RandomCrop((256, 768)) # KITTI
            #crop = RandomCrop((256, 384), augment=self.augment) # KITTI
            #crop = RandomCrop((512, 512), augment=self.augment) # girl 1K
            #crop = RandomCrop((1024, 1024), augment=self.augment) # girl 2K
            #crop = RandomCrop((384, 768)) # flyingthing, monkaa, driving
            #crop = RandomCrop((256, 768)) # KITTI
            #crop = RandomCrop((512, 512)) # girl 1k
            #crop = RandomCrop((1024, 1024)) # girl 2k
            #crop = RandomCrop((384, 768))
            #crop = RandomCrop((896, 896))

            sample = crop(sample)
            pass
        return sample

