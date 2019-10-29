from dataloader.EXRloader import load_exr
from utils.preprocess import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from losses.normalloss import angle_diff_norm 
import array
import OpenEXR

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

def make_grid(h, w):
    x = np.arange(0, w)
    y = np.arange(0, h)
    x_coord, y_coord = np.meshgrid(x, y)

    x_coord = torch.from_numpy(x_coord).float()
    y_coord = torch.from_numpy(y_coord).float()

    x_coord = w * 0.5 - x_coord
    y_coord = h * 0.5 - y_coord

    x_coord = torch.unsqueeze(x_coord, 0)
    y_coord = torch.unsqueeze(y_coord, 0)

    return x_coord, y_coord

def disp2norm(disp, coord, focus_length):
    if len(disp.shape) == 2:
        disp = np.expand_dims(disp, axis=0)
    c, h, w = disp.size()

    torch_disp = disp 

    dx = (torch_disp[ :, :, :-2] - torch_disp[ :, :, 2:]) * 0.5
    dy = (torch_disp[ :, :-2, :] - torch_disp[ :, 2:, :]) * 0.5

    #dx = (torch_disp[:, 1:-1, :-2] - torch_disp[:, 1:-1, 2:]) * 2.0 + (torch_disp[:, :-2, :-2] - torch_disp[:, :-2, 2:]) + (torch_disp[:, 2:, :-2] - torch_disp[:, 2:, 2:]) 
    #dy = (torch_disp[:, :-2, 1:-1] - torch_disp[:, 2:, 1:-1]) * 2.0 + (torch_disp[:, :-2, :-2] - torch_disp[:, 2:, :-2]) + (torch_disp[:, :-2, 2:] - torch_disp[:, 2:, 2:])

    #dx = dx / 8
    #dy = dy / 8

    nx = 1050.0 * dx / torch.abs(torch_disp[:,  :, 1:-1] - coord[0][ :, :, 1:-1] * dx)
    ny = 1050.0 * dy / torch.abs(torch_disp[:,  1:-1, :] - coord[1][ :, 1:-1, :] * dy)
    #nx = dx / torch.abs(torch_disp[:, 1:-1, 1:-1] - coord[0][:, 1:-1, 1:-1] * dx)
    #ny = dy / torch.abs(torch_disp[:, 1:-1, 1:-1] - coord[1][:, 1:-1, 1:-1] * dy)
    nz = -torch.ones(1,  h, w).float()

    nx = torch.squeeze(nn.functional.pad(torch.unsqueeze(nx, 0), (1, 1, 0, 0), 'replicate'), 0)
    ny = torch.squeeze(nn.functional.pad(torch.unsqueeze(ny, 0), (0, 0, 1, 1), 'replicate'), 0)
    #nx = torch.squeeze(nn.functional.pad(torch.unsqueeze(nx, 0), (1, 1, 1, 1), 'replicate'), 0)
    #ny = torch.squeeze(nn.functional.pad(torch.unsqueeze(ny, 0), (1, 1, 1, 1), 'replicate'), 0)
    #nx = F.pad(nx, (1, 1, 0, 0), 'replicate')
    #ny = F.pad(ny, (0, 0, 1, 1), 'replicate')

    norm = torch.cat((nx, ny, nz), dim=0) 

    norm = norm / torch.norm(norm, 2, dim=0, keepdim=True)

    return norm

#disp = load_exr("disp.exr")
disp, scale = load_pfm("disp.pfm")
disp = np.flip(disp, axis=0).copy()
print disp.shape, np.min(disp), np.max(disp)
gt_norm = load_exr("2.exr")

gt_norm = gt_norm * 2.0 - 1.0

disp = torch.from_numpy(disp).float()
gt_norm = torch.from_numpy(gt_norm.transpose([2, 0, 1])).float()
print disp.size()
print gt_norm.size()

disp = disp.unsqueeze(0)

c, h, w = disp.size()
coord = make_grid(h, w)

trans_norm = disp2norm(disp, coord, 1050.0)
trans_norm = (trans_norm + 1) * 0.5
print trans_norm.size()
save_exr(trans_norm.numpy(), 'trans_normal.exr')
trans_norm = trans_norm.unsqueeze(0)
trans_norm = trans_norm * 2.0 - 1.0
gt_norm = gt_norm.unsqueeze(0)
print F.l1_loss(gt_norm, trans_norm, size_average=True)
angle_diffs = angle_diff_norm(gt_norm, trans_norm)

print torch.min(angle_diffs), torch.max(angle_diffs), torch.mean(angle_diffs)


