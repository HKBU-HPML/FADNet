import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from utils import preprocess 
from skimage import transform, io
from dataloader.commons import normalize_method

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')
    #return io.imread(path)

def disparity_loader(path):
    return Image.open(path)
    #return Image.open(path)


class myImageFolder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader, normalize=normalize_method):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

        self.scale_size = (384, 1280)
        self.img_size = (384, 1280)
        self.normalize = normalize

    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size

    def __getitem__(self, index):
        left_fn  = self.left[index]
        right_fn = self.right[index]
        disp_L_fn = self.disp_L[index]

        left = np.array(self.loader(left_fn))
        right = np.array(self.loader(right_fn))
        dataL = self.dploader(disp_L_fn)

        h, w, _ = left.shape
        th, tw = 256, 512

        if self.normalize == 'imagenet':
           processed = preprocess.get_transform(augment=False)  
           img_left   = processed(left)
           img_right  = processed(right)
        else:
           img_left = np.zeros([3, h, w], 'float32')
           img_right = np.zeros([3, h, w], 'float32')
           for c in range(3):
               img_left[c, :, :] = (left[:, :, c] - np.mean(left[:, :, c])) / np.std(left[:, :, c])
               img_right[c, :, :] = (right[:, :, c] - np.mean(right[:, :, c])) / np.std(right[:, :, c])

        if self.training:  
 
           y1 = random.randint(0, h - th)
           x1 = random.randint(0, w - tw)

           img_left = img_left[:, y1:y1 + th, x1:x1 + tw]
           img_right = img_right[:, y1:y1 + th, x1:x1 + tw]

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]
           dataL = dataL[np.newaxis, :]

           sample = {  'img_left': img_left, 
                       'img_right': img_right, 
                       'img_names': [self.left[index], self.right[index], self.disp_L[index]],
                       'gt_disp': dataL
                    }

           #return left_img, right_img, dataL
           return sample

        else:
           top_pad = 384-h
           left_pad = 1280-w
           img_left = np.lib.pad(img_left,((0,0),(top_pad,0),(left_pad, 0)),mode='constant',constant_values=0)
           img_right = np.lib.pad(img_right,((0,0),(top_pad,0),(left_pad, 0)),mode='constant',constant_values=0)

           dataL = dataL.crop((w-1280, h-384, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[np.newaxis, :]

           sample = {  'img_left': img_left, 
                       'img_right': img_right, 
                       'img_names': [self.left[index], self.right[index], self.disp_L[index]],
                       'gt_disp': dataL
                    }

           #return left_img, right_img, dataL
           return sample

    def __len__(self):
        return len(self.left)
