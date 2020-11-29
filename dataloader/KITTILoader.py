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


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

        #self.scale_size = (384, 1280)
        self.scale_size = (1280, 384)

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        #origin_width, origin_height = left_img.size
        #scale_width = self.scale_size[0]
        #print("ori_w: %d, sca_w: %d" % (origin_width, scale_width))
 
        #left_img = left_img.resize(self.scale_size, Image.BILINEAR)
        #right_img = right_img.resize(self.scale_size, Image.BILINEAR)       
        #dataL = dataL.resize(self.scale_size, Image.BILINEAR)
        #left_img = transform.resize(left_img, self.scale_size, preserve_range=True)
        #right_img = transform.resize(right_img, self.scale_size, preserve_range=True)
        #dataL = transform.resize(dataL, self.scale_size, preserve_range=True) * 1.0 * scale_width / origin_width

        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
           #th, tw = 256, 896
           #th, tw = 384, 768
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(h // 4, h - th)
           #y1 = h - th
           #y1 = random.randint((h-th)/2, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           #dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256 * 1.0 * scale_width / origin_width
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)  
           #processed = preprocess.get_transform(augment=True)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)
           #print('[index:%d]left: %s, rect(%d,%d,%d,%d)'%(index, self.left[index], x1,y1,x1+tw,y1+th))

           return left_img, right_img, dataL
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-1280, h-384, w, h))
           right_img = right_img.crop((w-1280, h-384, w, h))
           w1, h1 = left_img.size

           dataL = dataL.crop((w-1280, h-384, w, h))
           #dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256 * 1.0 * scale_width / origin_width
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

           processed = preprocess.get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
