from __future__ import absolute_import, division, print_function
import torch
from torch.nn.functional import pad
import dataset
from skimage import io
import numpy as np
from layers_package.resample2d_package.modules.resample2d import Resample2d
import math

def validate_disparity(disp, left, right):
    disp = np.transpose(disp.numpy()[0], [1, 2, 0])
    left = np.transpose(left.numpy()[0], [1, 2, 0])
    right = np.transpose(right.numpy()[0], [1, 2, 0])
    val_right = np.zeros(right.shape, dtype=left.dtype)
    shape = right.shape
    for i in range(shape[0]):
        for j in reversed(range(shape[1])):
            d = int(math.ceil(disp[i][j][0]))
            #print('disp: ', d)
            right_j = j-d
            if right_j < 0 or right_j >= shape[1]:
                continue
            val_right[i][right_j] = left[i][j]
    val_right = np.transpose(val_right, [2, 0, 1])
    return val_right

def apply_disparity(input_images, x_offset, wrap_mode='border', tensor_type = 'torch.cuda.FloatTensor'):
    num_batch, num_channels, height, width = input_images.size()
    #print(input_images.size())
    #print(x_offset.size())

    # Handle both texture border types
    edge_size = 0
    if wrap_mode == 'border':
        edge_size = 1
        # Pad last and second-to-last dimensions by 1 from both sides
        input_images = pad(input_images, (1, 1, 1, 1))
    elif wrap_mode == 'edge':
        edge_size = 0
    else:
        return None

    # Put channels to slowest dimension and flatten batch with respect to others
    input_images = input_images.permute(1, 0, 2, 3).contiguous()
    im_flat = input_images.view(num_channels, -1)

    # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
    # meshgrid function)
    x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type)
    y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type)
    # Take padding into account
    x = x + edge_size
    y = y + edge_size
    # Flatten and repeat for each image in the batch
    x = x.view(-1).repeat(1, num_batch)
    y = y.view(-1).repeat(1, num_batch)

    # Now we want to sample pixels with indicies shifted by disparity in X direction
    # For that we convert disparity from % to pixels and add to X indicies
    x = x + x_offset.contiguous().view(-1) * width
    #x = x + x_offset.contiguous().view(-1) # direct predict disparity
    # Make sure we don't go outside of image
    x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
    # Round disparity to sample from integer-valued pixel grid
    y0 = torch.floor(y)
    # In X direction round both down and up to apply linear interpolation
    # between them later
    x0 = torch.floor(x)
    x1 = x0 + 1
    # After rounding up we might go outside the image boundaries again
    x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

    # Calculate indices to draw from flattened version of image batch
    dim2 = (width + 2 * edge_size)
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    # Set offsets for each image in the batch
    base = dim1 * torch.arange(num_batch).type(tensor_type)
    base = base.view(-1, 1).repeat(1, height * width).view(-1)
    # One pixel shift in Y  direction equals dim2 shift in flattened array
    base_y0 = base + y0 * dim2
    # Add two versions of shifts in X direction separately
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    # Sample pixels from images
    pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
    pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

    # Apply linear interpolation to account for fractional offsets
    weight_l = x1 - x
    weight_r = x - x0
    output = weight_l * pix_l + weight_r * pix_r

    # Reshape back into image batch and permute back to (N,C,H,W) shape
    output = output.view(num_channels, num_batch, height, width).permute(1,0,2,3)

    return output

if __name__ == '__main__':
    img_left_name = "/media/external/data/virtual2/ep0010/camera1_R/XNCG_ep0010_cam01_rd_lgt.0003.png"
    img_right_name = "/media/external/data/virtual2/ep0010/camera1_L/XNCG_ep0010_cam01_rd_lgt.0003.png"
    gt_disp_name = "/media/external/data/virtual2/ep0010/camera1_R/XNCG_ep0010_cam01_rd_lgt.Z.0003.pfm"
    #img_left_name = "./test_imgs/flying_left.png"
    #img_right_name = "./test_imgs/flying_right.png"
    #gt_disp_name = "./test_imgs/flying_left.pfm"

    gt_disp, scale = dataset.load_pfm(gt_disp_name)
    gt_disp = gt_disp[::-1, :]
    img_left = io.imread(img_left_name)[:, :, :3]
    img_right = io.imread(img_right_name)[:, :, :3]

    print(np.mean(gt_disp), np.mean(img_left), np.mean(img_right))

    img_left = np.transpose(img_left, [2, 0, 1])
    img_right = np.transpose(img_right, [2, 0, 1])

    gt_disp = torch.from_numpy(gt_disp[np.newaxis, np.newaxis, :].astype(np.float32))
    img_left = torch.from_numpy(img_left[np.newaxis, :].astype(np.float32))
    img_right = torch.from_numpy(img_right[np.newaxis, :].astype(np.float32))

    print(gt_disp.size(), img_left.size(), img_right.size())

    # shaohuai's warp
    warp_left = validate_disparity(gt_disp, img_left, img_right)

    # resample warp
    #resample1 = Resample2d()
    #dummy_flow = torch.autograd.Variable(torch.zeros(img_right.size()))
    #gt_disp = torch.cat((gt_disp, dummy_flow), dim = 1)
    #warp_left = resample1(img_right.cuda(), -gt_disp.cuda()).data.cpu()
    #warp_left = warp_left.numpy()[0]

    # monodepth warp
    #warp_left = apply_disparity(img_right, -gt_disp, tensor_type = 'torch.FloatTensor')
    #warp_left = warp_left.numpy()[0]

    print(warp_left.shape)
    print(np.mean(warp_left))
    io.imsave("test_reverse.png", np.transpose(warp_left, [1, 2, 0]).astype(np.int))

