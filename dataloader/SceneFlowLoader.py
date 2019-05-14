from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from utils.preprocess import *
from torchvision import transforms

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
        self.phase = phase
        self.scale_size = (576, 960)
        #self.scale_size = (640, 1024)
        #self.scale_size = (768, 1280)
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

            img = None
            if filename.find('.npy') > 0:
                img = np.load(filename)
            else:
                img = io.imread(filename)
            return img
           
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

        if self.phase == 'detect' or self.phase == 'test':
            img_left = transform.resize(img_left, self.scale_size, preserve_range=True)
            img_right = transform.resize(img_right, self.scale_size, preserve_range=True)

            # change image pixel value type ot float32
            img_left = img_left.astype(np.float32)
            img_right = img_right.astype(np.float32)
            #scale = RandomRescale((1024, 1024))
            #sample = scale(sample)

        if self.phase == 'detect' or self.phase == 'test':
            rgb_transform = default_transform()
        else:
            rgb_transform = inception_color_preproccess()

        img_left = rgb_transform(img_left)
        img_right = rgb_transform(img_right)

        gt_disp = gt_disp[np.newaxis, :]
        gt_disp = torch.from_numpy(gt_disp.copy()).float()

        if self.phase == 'train':

            h, w = img_left.shape[1:3]
            th, tw = 384, 768
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)

            img_left = img_left[:, top: top + th, left: left + tw]
            img_right = img_right[:, top: top + th, left: left + tw]
            gt_disp = gt_disp[:, top: top + th, left: left + tw]

        #elif self.phase == 'test':

        #    img_left = img_left[:, :512, :]
        #    img_right = img_right[:, :512, :]
        #    gt_disp = gt_disp[:, :512, :]


        sample = {'img_left': img_left, 
                  'img_right': img_right, 
                  'gt_disp' : gt_disp,   
                  'img_names': img_names
                 }


        return sample

