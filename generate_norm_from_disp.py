
import numpy as np
import OpenEXR
import Imath
import imageio
import torch
import torch.nn as nn
import re
import array
from PIL import Image

import glob
import os
import sys
import argparse



def exr2hdr(exrpath):
	File = OpenEXR.InputFile(exrpath)
	PixType = Imath.PixelType(Imath.PixelType.FLOAT)
	DW = File.header()['dataWindow']
	CNum = len(File.header()['channels'].keys())
	if (CNum > 1):
		Channels = ['R', 'G', 'B']
		CNum = 3
	else:
		Channels = ['G']
	Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
	Pixels = [numpy.fromstring(File.channel(c, PixType), dtype=numpy.float32) for c in Channels]
	hdr = numpy.zeros((Size[1],Size[0],CNum),dtype=numpy.float32)
	if (CNum == 1):
		hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
	else:
		hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
		hdr[:,:,1] = numpy.reshape(Pixels[1],(Size[1],Size[0]))
		hdr[:,:,2] = numpy.reshape(Pixels[2],(Size[1],Size[0]))
	return hdr


def save_exr(img, filename):
	c, h, w = img.shape
	if c == 1:
		Gs = array.array('f', img).tostring()

		out = OpenEXR.OutputFile(exrpath, OpenEXR.Header(w, h))
		out.writePixels({'G' : Gs })
	else:
		data = np.array(img).reshape(c, w * h)
                Rs = array.array('f', data[0,:]).tostring()
                Gs = array.array('f', data[1,:]).tostring()
                Bs = array.array('f', data[2,:]).tostring()

		out = OpenEXR.OutputFile(filename, OpenEXR.Header(w, h))
		out.writePixels({'R' : Rs, 'G' : Gs, 'B' : Bs })


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

def load_disp(filename):
    disp = None
    if filename.endswith('pfm'):
        disp, scale = load_pfm(filename)
        disp = disp[::-1, :].copy()
    elif filename.endswith('exr'):
        disp = exr2hdr(filename)
    else:
        f_in = np.array(Image.open(filename))
        d_r = f_in[:,:,0].astype('float32')
        d_g = f_in[:,:,1].astype('float32')
        d_b = f_in[:,:,2].astype('float32')

        disp = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    return disp


def make_grid(size):
    x = np.arange(0, size[1])
    y = np.arange(0, size[0])
    x_coord, y_coord = np.meshgrid(x, y)

    x_coord = torch.from_numpy(x_coord).cuda().float()
    y_coord = torch.from_numpy(y_coord).cuda().float()

    x_coord = size[1] * 0.5 - x_coord
    y_coord = size[0] * 0.5 - y_coord

    x_coord = torch.unsqueeze(x_coord, 0)
    y_coord = torch.unsqueeze(y_coord, 0)

    return x_coord, y_coord


def disp2norm(disp, coord, focus_length):
    if len(disp.shape) == 2:
        disp = np.expand_dims(disp, axis=0)
    c, h, w = disp.shape

    torch_disp = torch.from_numpy(disp).cuda() 

    dx = (torch_disp[:, :, :-2] - torch_disp[:, :, 2:]) * 0.5
    dy = (torch_disp[:, :-2, :] - torch_disp[:, 2:, :]) * 0.5

    #dx = (torch_disp[:, 1:-1, :-2] - torch_disp[:, 1:-1, 2:]) * 2.0 + (torch_disp[:, :-2, :-2] - torch_disp[:, :-2, 2:]) + (torch_disp[:, 2:, :-2] - torch_disp[:, 2:, 2:]) 
    #dy = (torch_disp[:, :-2, 1:-1] - torch_disp[:, 2:, 1:-1]) * 2.0 + (torch_disp[:, :-2, :-2] - torch_disp[:, 2:, :-2]) + (torch_disp[:, :-2, 2:] - torch_disp[:, 2:, 2:])

    #dx = dx / 8
    #dy = dy / 8

    nx = dx / torch.abs(torch_disp[:, :, 1:-1] - coord[0][:, :, 1:-1] * dx)
    ny = dy / torch.abs(torch_disp[:, 1:-1, :] - coord[1][:, 1:-1, :] * dy)
    #nx = dx / torch.abs(torch_disp[:, 1:-1, 1:-1] - coord[0][:, 1:-1, 1:-1] * dx)
    #ny = dy / torch.abs(torch_disp[:, 1:-1, 1:-1] - coord[1][:, 1:-1, 1:-1] * dy)
    nz = -torch.ones(1, h, w).cuda() / focus_length

    nx = torch.squeeze(nn.functional.pad(torch.unsqueeze(nx, 0), (1, 1, 0, 0), 'replicate'), 0)
    ny = torch.squeeze(nn.functional.pad(torch.unsqueeze(ny, 0), (0, 0, 1, 1), 'replicate'), 0)
    #nx = torch.squeeze(nn.functional.pad(torch.unsqueeze(nx, 0), (1, 1, 1, 1), 'replicate'), 0)
    #ny = torch.squeeze(nn.functional.pad(torch.unsqueeze(ny, 0), (1, 1, 1, 1), 'replicate'), 0)

    norm = torch.cat((nx, ny, nz), dim=0) 

    norm = norm / torch.norm(norm, 2, dim=0, keepdim=True)

    norm = norm.cpu().numpy()
    
    return norm


def get_files_list_with_filter(directory_path, filt):
    files = glob.glob(directory_path + filt)
    if len(files) == 0: #search sub directories
        dir_filter = directory_path + '*' 
        dirs = glob.glob(dir_filter)
        for _dir in dirs:
            if os.path.isdir(_dir):
                _dir = _dir.replace('\\', '/') + '/'
                _list = get_files_list_with_filter(_dir, filt)
                files = files + _list
    return files

def generate_normals(opt):
    if opt.dataset == 'FlyingThings':
        ext = '*.pfm'
    elif opt.dataset == 'Sintel':
        ext = '*.png'
    else:
        ext = '*.exr'

    files = get_files_list_with_filter(opt.path, ext)
    disp =  load_disp(files[0])
    if len(disp.shape) == 2:
        size = disp.shape
    else:
        c, h, w = disp.shape
        size = [h, w]
    print ('img size: ', size)
    coord = make_grid(size)

    total_num = len(files)
    count = 0
    for filename in files:
        disp = load_disp(filename)
        norm = disp2norm(disp, coord, opt.focus_length)

        norm = (norm + 1) * 0.5
        if opt.norm_path == '':
            out_file = os.path.dirname(filename) + '/' + os.path.basename(filename).split('.')[0] + '.exr'
        else:
            relative_dir = os.path.relpath(os.path.dirname(filename), opt.path)
            out_path = opt.norm_path + '/' + relative_dir + '/'
            out_file = out_path + os.path.basename(filename).split('.')[0] + '.exr'
            if not os.path.isdir(out_path):
                os.makedirs(out_path)

        save_exr(norm, out_file)

        count = count + 1
        print ('[', count, '/', total_num, '] ', out_file) 
       


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='the disparity folder path', default='')
    parser.add_argument('--norm_path', type=str, help='the output of surface normal folder path', default='')
    parser.add_argument('--focus_length', type=float, help='camera focus length', default=600)
    parser.add_argument('--dataset', type=str, help='dispairties from which dataset', default='FlyingThings')

    opt = parser.parse_args()
    if opt.path == '':
        print ('No valid path of disparity')
        exit()

    generate_normals(opt)
