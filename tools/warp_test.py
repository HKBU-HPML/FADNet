from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
from utils.preprocess import *
from skimage import io, transform
from torch.autograd import Variable

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    
    mask[mask<0.999] = 0
    mask[mask>0] = 1
    
    return output*mask

img = io.imread("tools/right.png").astype(np.float32)
img = torch.from_numpy(img.copy()).float()
img = img.permute(2, 0, 1)
img = img.unsqueeze(0).cuda()

left_img = io.imread("tools/left.png").astype(np.float32)
left_img = torch.from_numpy(left_img.copy()).float()
left_img = left_img.permute(2, 0, 1)
left_img = left_img.unsqueeze(0).cuda()

disp, scale = load_pfm("tools/left.pfm")
disp = disp[::-1, :]
disp = disp[np.newaxis, :]
disp = torch.from_numpy(disp.copy()).float()
dummy_y = torch.zeros(disp.size())
flo = torch.cat((-disp, dummy_y), dim = 0)
flo = flo.unsqueeze(0).cuda()

print(img.size(), flo.size(), img.mean())

output = warp(img, flo)
err_map = abs(output - left_img)
print(output.mean())

output = output.squeeze(0).permute(1, 2, 0)
err_map = err_map.squeeze(0).permute(1, 2, 0)

io.imsave("tools/warp_left.png",(output*256).cpu().numpy().astype('uint16'))
io.imsave("tools/err_left.png",(err_map*256).cpu().numpy().astype('uint16'))



