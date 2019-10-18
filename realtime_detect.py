from __future__ import print_function

import argparse
import os
import time

import cv2
import pyzed.sl as sl
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image

from utils.preprocess import default_transform, scale_transform, __imagenet_stats
from skimage import transform

from detecter import init_net, init_data, detect_batch

# point cloud visualization
import open3d as o3d

ZED_cam_bl = 0.0
ZED_cam_fl = 0.0
ZED_cam_res = (0.0, 0.0)

def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_resolution().width, 2), cam.get_resolution().height))
    print("Focus Length: {0}, Baseline: {1}.".format(round(ZED_cam_fl, 2), round(ZED_cam_bl, 2)))
    print("Camera FPS: {0}.".format(cam.get_camera_fps()))
    print("Firmware: {0}.".format(cam.get_camera_information().firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))


def print_help():
    print("Help for camera setting controls")
    print("  Quit:                               q\n")

def init_vis(size=(960,540), light_on=False, point_size=0.1, background=np.asarray([0, 0, 0])):
    vis = o3d.visualization.Visualizer()
    vis.create_window('Point Cloud', size[0], size[1])
    pcd = o3d.geometry.PointCloud()
    p = np.random.rand(960 * 540, 3)
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(p)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = background
    opt.light_on = light_on
    opt.point_size = point_size

    return vis, pcd

def init_live_camera():
    print("Setup camera")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    global ZED_cam_fl, ZED_cam_bl, ZED_cam_res
    cali = cam.get_camera_information().calibration_parameters
    ZED_cam_bl = cali.T[0] / 1000.0
    ZED_cam_fl = cali.left_cam.fx
    ZED_cam_res = (cam.get_resolution().width, cam.get_resolution().height)

    print_camera_information(cam)
    print_help()

    return cam

import time
g_time = 0
def time_cost(title = None):
    global g_time
    current_time = time.time()
    if title is not None:
        print (title, ' ', current_time - g_time)
    g_time = current_time


def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img,volatile=True), size)).data

def transform_image(img, stack=True):
    #time_cost()
    img = img[:,:,:3]
    #time_cost('channel change:')
    img = img.astype(np.float32)
    #time_cost('float32:')
    #img = Image.fromarray(img)

    rgb_transform = default_transform()
    img = rgb_transform(img)
    #time_cost('tensor transform:')

    img = resize2d(img, (576, 960))
    #time_cost('resize:')

    if stack:
        img = torch.stack([img], 0)
    #time_cost('stack:')
    return img

def transform_images(imgs):
    batch = []
    for img in imgs:
        t = transform_image(img, False)
        batch = batch + [t]
    imgs = torch.stack(batch, 0)
    return imgs
    

def make_batch(left, right):
    left = transform_image(left)
    right = transform_image(right)
    batch = {
        'img_left' : left,
        'img_right' : right,
    }
    return batch

def make_batch_by_imgs(imgs):
    imgs = np.asarray(imgs)
    left = imgs[:, 0]
    right = imgs[:, 1]
    imgs = np.concatenate((left, right), axis=0)
    imgs = transform_images(imgs)
    batch = {
        'img_left' : imgs[::2],
        'img_right' : imgs[1::2],
    }
    return batch

def get_point_cloud(left_img, disparity_map, baseline, focus_length, scale=1.0, limit=0.1, pcd=None):
    time_cost()
    size = disparity_map.shape
    disparity_map[disparity_map < limit] = limit
    depth = (baseline * focus_length / disparity_map)
    time_cost('depth calc:')

    y = np.linspace(-0.5, 0.5, size[0])
    x = np.linspace(-0.5, 0.5, size[1])
    xv, yv = np.meshgrid(x, y)
    time_cost('mesh grid:')

    # X, Y, Z, R, G, B
    pc = np.zeros((size[0], size[1], 3), dtype=np.float64)
    pc[:,:,0] = xv * depth * (size[1] / focus_length * scale)
    pc[:,:,1] = yv * depth * (-size[0] / focus_length * scale)
    pc[:,:,2] = depth * (-scale)
    time_cost('pc calc:')

    pc = np.reshape(pc, (size[0]*size[1], 3))
    rgb = np.reshape(left_img[:,:,:3], (size[0]*size[1], 3)) / 255.0
    time_cost('reshape:')

    if pcd is None:
        pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    time_cost('pcd =:')

    return pcd


def PCVisualization(vis, point_cloud):
    #print (point_cloud)
    #draw_geometries([point_cloud], 'PointCloud', 960, 540)
    #draw_geometries_with_editing([point_cloud], 'PointCloud', 960, 540)

    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    #vis.run()
    time_cost('vis:')

    print ('')


def simple_loop(cam, left, right, net, net_name):
    time_cost()
    cam.retrieve_image(left, sl.VIEW.VIEW_LEFT)
    left_img = left.get_data()
    cv2.imshow("ZED_Left", left_img)
    time_cost('left img:')
   
    cam.retrieve_image(right, sl.VIEW.VIEW_RIGHT)
    right_img = right.get_data()
    cv2.imshow("ZED_Right", right_img)
    time_cost('right img:')
    
    batch = make_batch(left_img, right_img)
    torch.cuda.synchronize()
    time_cost('make_batch:')

    output, input_var = detect_batch(net, batch, net_name, (540, 960))
    torch.cuda.synchronize()
    time_cost('detect:')

    depth = output[0][0].data.cpu().numpy()
    torch.cuda.synchronize()
    time_cost('to cpu:')
    depth = depth / 255
    time_cost('divide 255:')
    cv2.imshow(net_name, depth)
    time_cost('show depth:')

    print('')

def point_cloud_loop(cam, left, right, net, net_name, vis, pcd):
    cam.retrieve_image(left, sl.VIEW.VIEW_LEFT)
    left_img = left.get_data()
    left_img = cv2.resize(left_img, (960, 540), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("ZED_Left", left_img)
   
    cam.retrieve_image(right, sl.VIEW.VIEW_RIGHT)
    right_img = right.get_data()
    right_img = cv2.resize(right_img, (960, 540), interpolation=cv2.INTER_LINEAR)
    
    batch = make_batch(left_img, right_img)
    torch.cuda.synchronize()

    output, input_var = detect_batch(net, batch, net_name, (540, 960))
    torch.cuda.synchronize()

    depth = output[0][0].data.cpu().numpy()
    torch.cuda.synchronize()
    cv2.imshow(net_name, depth / 255)

    point_cloud = get_point_cloud(left_img, depth, ZED_cam_bl, ZED_cam_fl, 1.0, pcd=pcd)
    PCVisualization(vis, point_cloud)



cache = []

def delay_loop(cam, left, right, net, net_name, delay_frame=1):
    time_cost()
    cam.retrieve_image(left, sl.VIEW.VIEW_LEFT)
    left_img = left.get_data()
    time_cost('left img:')
   
    cam.retrieve_image(right, sl.VIEW.VIEW_RIGHT)
    right_img = right.get_data()
    time_cost('right img:')
    
    batch = make_batch(left_img, right_img)
    time_cost('make_batch:')

    output, input_var = detect_batch(net, batch, net_name)
    time_cost('detect:')

    depth = output[0][0].data / 255
    time_cost('divide 255:')

    global cache
    if len(cache) >= delay_frame:
        imgs = cache[0]
        cv2.imshow("ZED_Left", imgs[0])
        cv2.imshow("ZED_Right", imgs[1])
        cv2.imshow(net_name, imgs[2].cpu().numpy())
        time_cost('cpu & show:')
        cache = cache[1:]
    cache = cache + [[left_img, right_img, depth]]
    time_cost('cache :')

    print('')

prepared_imgs = []
processed_imgs = []

def batch_loop(cam, left, right, net, net_name, batch_size):
    global prepared_imgs
    global processed_imgs

    cam.retrieve_image(left, sl.VIEW.VIEW_LEFT)
    left_img = left.get_data()
    cam.retrieve_image(right, sl.VIEW.VIEW_RIGHT)
    right_img = right.get_data()
    prepared_imgs = prepared_imgs + [[left_img, right_img]]

    if len(prepared_imgs) >= batch_size:
        batch = make_batch_by_imgs(prepared_imgs)
        output, input_var = detect_batch(net, batch, net_name)

        depth = output[:][0].data / 255
        depth = depth.cpu().numpy()
        for i in range(len(depth)):
            processed_imgs = processed_imgs + [[prepared_imgs[i][0], prepared_imgs[i][1], depth[i]]]
        prepared_imgs = []

    if len(processed_imgs) > 0:
        imgs = processed_imgs[0]
        processed_imgs = processed_imgs[1:]

        cv2.imshow("ZED_Left", imgs[0])
        cv2.imshow("ZED_Right", imgs[1])
        cv2.imshow(net_name, imgs[2])



def realtime_detect(opt):
    vis, pcd = init_vis()
    cam = init_live_camera()

    runtime = sl.RuntimeParameters()
    left = sl.Mat()
    right = sl.Mat()

    devices = [int(item) for item in opt.devices.split(',')]
    net = init_net(opt.model, devices, opt.net)

    # loop until keyboard-q
    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            a = time.time()

            #simple_loop(cam, left, right, net, opt.net)
            #delay_loop(cam, left, right, net, opt.net, 2)
            #batch_loop(cam, left, right, net, opt.net, 1)
            point_cloud_loop(cam, left, right, net, opt.net, vis, pcd)
            
            a = time.time() - a
            print ('TOTAL COST: ', a)
            print ('')

            
        key = cv2.waitKey(1)

    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='dispnetcres')
    parser.add_argument('--model', type=str, help='model to load', default='best.pth')
    parser.add_argument('--devices', type=str, help='devices', default='0')

    opt = parser.parse_args()
    realtime_detect(opt)