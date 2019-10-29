from __future__ import print_function

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image

from utils.preprocess import default_transform, scale_transform, __imagenet_stats
from skimage import io, transform

from detecter import init_net, init_data, detect_batch
from realtime_detect import init_vis, make_batch, get_point_cloud, PCVisualization

# point cloud visualization
import open3d as o3d


left_img_file = 'data/3drecon_test/l_1.png'
right_img_file = 'data/3drecon_test/r_1.png'
baseline = 0.1
focus_length = 480
scale = 0.2


def load_img(path):
	img = io.imread(path)
	img = img[:,:,:3]
	return img


def get_mesh_from_pcd(pcd):
	radii = [1, 2]
	mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
	return mesh


def reconstruction(opt):
    vis, pcd = init_vis()

    devices = [int(item) for item in opt.devices.split(',')]
    net = init_net(opt.model, devices, opt.net)

    left_img = load_img(left_img_file)
    right_img = load_img(right_img_file)

    show_left = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Left", show_left)

    batch = make_batch(left_img, right_img)
    torch.cuda.synchronize()

    output, input_var = detect_batch(net, batch, opt.net, (540, 960))
    torch.cuda.synchronize()

    disp = output[0, 3, :, :].data.cpu().numpy()
    normal = output[0, :3, :, :].data.cpu().numpy()
    torch.cuda.synchronize()
    cv2.imshow(opt.net, disp / 255)

    pcd = get_point_cloud(left_img, disp, baseline, focus_length, scale, 3, pcd)
    #PCVisualization(vis, point_cloud)

    '''
    size = left_img.shape
    print (size)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(size[1], size[0], focus_length, focus_length, int(size[1] / 2), int(size[0] / 2))
    disp[disp < 3] = 3
    depth = baseline * focus_length / disp
    rgb = o3d.geometry.Image(left_img.astype(np.uint8))
    depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    '''

    normals = normal.transpose([1, 2, 0])
    normals = np.reshape(normals, (540 * 960, 3))
    #pcd.colors = o3d.utility.Vector3dVector(normals)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    #vis.run()

    #print ('estimating normals.')
    #point_cloud.estimate_normals()
    '''
    print ('generating mesh.')
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    print (pcd)
    print (voxel_down_pcd)
    mesh = get_mesh_from_pcd(voxel_down_pcd)
    print (mesh)
    print ('finish mesh.')
    '''

    #o3d.visualization.draw_geometries([voxel_down_pcd])
    #vis.remove_geometry(pcd)
    #vis.add_geometry(voxel_down_pcd)
    #vis.add_geometry(mesh)
    vis.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='dispnetcres')
    parser.add_argument('--model', type=str, help='model to load', default='best.pth')
    parser.add_argument('--devices', type=str, help='devices', default='0')

    opt = parser.parse_args()
    reconstruction(opt)