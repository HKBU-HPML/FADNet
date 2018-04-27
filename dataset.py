from __future__ import print_function, division
import os, re, sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(filename):

  file = open(filename, 'r')
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().rstrip()
  if header == 'PF':
    color = True    
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  file.close()
  return np.reshape(data, shape), scale

'''
Save a Numpy array to a PFM file.
'''
def save_pfm(filename, image, scale = 1):
  file = open(filename, 'w')
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file) 
  file.close()

class RandomRescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_left, image_right, gt_disp = sample['img_left'], sample['img_right'], sample['gt_disp']
        h, w = image_left.shape[:2]
        #if isinstance(self.output_size, int):
        out_h, out_w = self.output_size
        #    if h > w:
        #        new_h, new_w = self.output_size * h / w, self.output_size
        #    else:
        #        new_h, new_w = self.output_size, self.output_size * w / h
        #else:
        #    new_h, new_w = self.output_size

        #new_h, new_w = int(new_h), int(new_w)

        image_left = transform.resize(image_left, self.output_size, preserve_range=True)
        image_right = transform.resize(image_right, self.output_size, preserve_range=True)

        # change image pixel value type ot float32
        image_left = image_left.astype(np.float32)
        image_right = image_right.astype(np.float32)
        gt_disp = gt_disp.astype(np.float32)
        new_sample = sample
        new_sample.update({'img_left': image_left, 
                      'img_right': image_right, 
                      'gt_disp': gt_disp})
        return new_sample

    @staticmethod
    def scale_back(disp, original_size=(1, 540, 960)):
        print('current shape:', disp.shape)
        o_w = original_size[2]
        s_w = disp.shape[2]
        trans_disp = transform.resize(disp, original_size, preserve_range=True)
        trans_disp = trans_disp * (o_w * 1.0 / s_w)
        print('trans shape:', trans_disp.shape)
        return trans_disp.astype(np.float32)


class RandomCrop(object):
    """
    Crop the image randomly
    Args: int or tuple. tuple is (h, w)

    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image_left, image_right, gt_disp = sample['img_left'], sample['img_right'], sample['gt_disp']

        h, w = image_left.shape[1:3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # top = 0
        # left = 0

        image_left = image_left[:, top: top + new_h, left: left + new_w]
        image_right = image_right[:, top: top + new_h, left: left + new_w]
        gt_disp = gt_disp[:, top: top + new_h, left: left + new_w]
        new_sample = sample
        new_sample.update({'img_left': image_left, 
                      'img_right': image_right, 
                      'gt_disp': gt_disp})

        return new_sample

class ToTensor(object):

    def __call__(self, array):
        # image_left, image_right, gt_disp = sample['img_left'], sample['img_right'], sample['gt_disp']

        # image_left = image_left.transpose((2, 0, 1))
        # image_right = image_right.transpose((2, 0, 1))
        # gt_disp = gt_disp[np.newaxis, :]

        # new_sample = {'img_left': torch.from_numpy(image_left), \
        #               'img_right': torch.from_numpy(image_right), \
        #               'gt_disp': torch.from_numpy(gt_disp.copy()) \
        #               }
        # return new_sample
        if len(array.shape) == 3 and array.shape[2] == 3:
            array = np.transpose(array, [2, 0, 1])
        if len(array.shape) == 2:
            array = array[np.newaxis, :]

        tensor = torch.from_numpy(array.copy())
        return tensor.float()

class DispDataset(Dataset):

    def __init__(self, txt_file, root_dir, transform = None, phase='train' ):
        """
        Args:
            txt_file [string]: Path to the image list
            transform (callable, optional): Optional transform to be applied                on a sample
        """
        f = open(txt_file, "r")
        self.imgPairs = f.readlines()
        f.close()
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.imgPairs)

    def __getitem__(self, idx):
        img_names = self.imgPairs[idx].split()
        img_left_name = os.path.join(self.root_dir, img_names[0])
        img_right_name = os.path.join(self.root_dir, img_names[1])
        gt_disp_name = os.path.join(self.root_dir, img_names[2])

        img_left = io.imread(img_left_name)
        img_right = io.imread(img_right_name)

        gt_disp = None
        scale = 1
        if os.path.isfile(gt_disp_name):
            gt_disp, scale = load_pfm(gt_disp_name)
            gt_disp = gt_disp[::-1, :]

        sample = {'img_left': img_left, 
                  'img_right': img_right, 
                  # 'pm_disp' : pm_disp,  
                  # 'pm_cost' : pm_cost,  
                  'gt_disp' : gt_disp,   
                  'img_names' : img_names
                 }

        if self.phase == 'test':
            #scale = RandomRescale((384, 768))
            scale = RandomRescale((1024, 1024))
        #    scale = RandomRescale((768, 1536))
        #    scale = RandomRescale((384, 768))
            #scale = RandomRescale((512 * 3, 896 * 3))
            #scale = RandomRescale((768, 1024 + 512))
            #scale = RandomRescale((1536, 1536))
            sample = scale(sample)

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
            #crop = RandomCrop((1024, 1024))
            sample = sample #crop(sample)
        return sample

