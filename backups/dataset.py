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
        self.transform = transform
        self.phase = phase
        self.augment = augment 
        self.center_crop = center_crop

    def __len__(self):
        return len(self.imgPairs)

    def __getitem__(self, idx):
        img_names = self.imgPairs[idx].rstrip().split()
        try:
            img_left_name = os.path.join(self.root_dir, img_names[0])
            img_right_name = os.path.join(self.root_dir, img_names[1])
            gt_disp_name = os.path.join(self.root_dir, img_names[2])
            ir_left_name = None
            ir_right_name = None
            if len(img_names) > 4:
                ir_left_name = os.path.join(self.root_dir, img_names[3])
                ir_right_name = os.path.join(self.root_dir, img_names[4])
            if img_left_name.find('.npy') > 0:
                img_left = np.load(img_left_name)
                img_right = np.load(img_right_name)
            else:
                img_left = io.imread(img_left_name)
                img_right = io.imread(img_right_name)
            if ir_left_name:
                if ir_left_name.find('.npy') > 0:
                    ir_left = np.load(ir_left_name)[:, :, 0]
                    ir_right= np.load(ir_right_name)[:, :, 0]
                else:
                    ir_left = io.imread(ir_left_name)[:, :, 0]
                    ir_right = io.imread(ir_right_name)[:, :, 0]
                #img_left = np.concatenate((img_left, ir_left[:, :, np.newaxis]), axis=2)
                #img_right = np.concatenate((img_right, ir_right[:, :, np.newaxis]), axis=2)
                with_ir_left = np.zeros(shape=(img_left.shape[0], img_left.shape[1], 4), dtype=ir_left.dtype)
                with_ir_right = np.zeros(shape=(img_left.shape[0], img_left.shape[1], 4), dtype=ir_left.dtype)
                with_ir_left[:,:,0:3] = img_left[:,:,0:3]
                with_ir_left[:,:,3] = ir_left
                with_ir_right[:,:,0:3] = img_right[:,:,0:3]
                with_ir_right[:,:,3] = ir_right
                img_right = with_ir_right
                img_left = with_ir_left
            else:
                img_left = img_left[:, :, 0:3]
                img_right = img_right[:, :, 0:3]
        except Exception as e:
            print('e: ', e, ' img_names: ', img_names)
            exit(1)

        gt_disp = None
        scale = 1
        # if os.path.isfile(gt_disp_name):
        if gt_disp_name.endswith('pfm'):
            gt_disp, scale = load_pfm(gt_disp_name)
            gt_disp = gt_disp[::-1, :]
        elif gt_disp_name.endswith('npy'):
            gt_disp = np.load(gt_disp_name)
            gt_disp = gt_disp[::-1, :]
        else:
            gt_disp = Image.open(gt_disp_name)
            gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256
        #if img_left.shape[0] != 2048 or img_right.shape[0] != 2048 or gt_disp.shape[0] != 2048:
        #    print('Error in shape: ', img_left.shape, img_right.shape, gt_disp.shape, img_names)

        sample = {'img_left': img_left, 
                  'img_right': img_right, 
                  # 'pm_disp' : pm_disp,  
                  # 'pm_cost' : pm_cost,  
                  'gt_disp' : gt_disp,   
                  'img_names' : img_names
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

