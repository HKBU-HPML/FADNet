from networks.submodules import *
from networks.EXRloader import exr2hdr
import torch
import torch.nn.functional as F


init_disp_file = '/test_norm2disp/init_disp.exr'
gt_disp_file = '/test_norm2disp/d_401.exr'
gt_norm_file = '/test_norm2disp/n_401.exr'



def EPE(gt_disp, disp):
	return F.smooth_l1_loss(gt_disp[target_valid], disp[target_valid], size_average=True)


init_disp = exr2hdr(init_disp_file)
init_disp = init_disp[::-1,:,:]
init_disp = init_disp.transpose([2, 0, 1])
init_disp = init_disp[np.newaxis, :, :, :]
init_disp = torch.from_numpy(init_disp).cuda()


gt_disp = exr2hdr(gt_disp_file)
gt_disp = gt_disp.transpose([2, 0, 1])
gt_disp = gt_disp[np.newaxis, :, :, :]
gt_disp = torch.from_numpy(gt_disp).cuda()


gt_norm = exr2hdr(gt_norm_file)
gt_norm = gt_norm.transpose([2, 0, 1])
gt_norm = gt_norm[np.newaxis, :, :, :]
gt_norm = torch.from_numpy(gt_norm).cuda()


print ('Ori EPE:', EPE(gt_disp, init_disp))


