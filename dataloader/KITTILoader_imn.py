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
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

        self.scale_size = (384, 1280)
        self.img_size = (384, 1280)

    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size

    def __getitem__(self, index):
        left_fn  = self.left[index]
        right_fn = self.right[index]
        disp_L_fn = self.disp_L[index]

        left_img = self.loader(left_fn)
        right_img = self.loader(right_fn)
        dataL = self.dploader(disp_L_fn)

        w, h = left_img.size
        th, tw = 256, 512

        if self.training:  
 
           y1 = random.randint(0, h - th)
           x1 = random.randint(0, w - tw)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]
           dataL = dataL[np.newaxis, :]

           sample = {  'img_left': left_img, 
                       'img_right': right_img, 
                       'img_names': [self.left[index], self.right[index], self.disp_L[index]],
                       'gt_disp': dataL
                    }

           return sample

        else:

           left_img = left_img.crop((w-1280, h-384, w, h))
           right_img = right_img.crop((w-1280, h-384, w, h))
           
           processed = preprocess.get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           dataL = dataL.crop((w-1280, h-384, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[np.newaxis, :]

           sample = {  'img_left': left_img, 
                       'img_right': right_img, 
                       'img_names': [self.left[index], self.right[index], self.disp_L[index]],
                       'gt_disp': dataL
                    }

           return sample

    def __len__(self):
        return len(self.left)
