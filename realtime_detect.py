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

from utils.preprocess import default_transform
from skimage import transform

from detecter import init_net, init_data, detect_batch

def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_resolution().width, 2), cam.get_resolution().height))
    print("Camera FPS: {0}.".format(cam.get_camera_fps()))
    print("Firmware: {0}.".format(cam.get_camera_information().firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))


def print_help():
    print("Help for camera setting controls")
    print("  Quit:                               q\n")

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

def transform_image(img):
    #time_cost()
    img = img[:,:,:3]
    #time_cost('channel change:')
    img = img.astype(np.float32)
    #time_cost('float32:')

    rgb_transform = default_transform()
    img = rgb_transform(img)
    #time_cost('tensor transform:')

    img = resize2d(img, (576, 960))
    #time_cost('resize:')

    img = torch.stack([img], 0)
    #time_cost('stack:')
    return img
    

def make_batch(left, right):
    left = transform_image(left)
    right = transform_image(right)
    batch = {
        'img_left' : left,
        'img_right' : right,
    }
    return batch

def realtime_detect(opt):
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
            cam.retrieve_image(left, sl.VIEW.VIEW_LEFT)
            left_img = left.get_data()
            cv2.imshow("ZED_Left", left_img)
           
            cam.retrieve_image(right, sl.VIEW.VIEW_RIGHT)
            right_img = right.get_data()
            cv2.imshow("ZED_Right", right_img)

            time_cost()
            batch = make_batch(left_img, right_img)
            time_cost('make_batch:')

            output, input_var = detect_batch(net, batch, opt.net, (540, 960))
            time_cost('detect:')

            depth = output[0][0].data.cpu().numpy()
            time_cost('to cpu:')
            depth = depth / 255
            time_cost('divide 255:')
            cv2.imshow(opt.net, depth)

        key = cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='dispnetcres')
    parser.add_argument('--model', type=str, help='model to load', default='best.pth')
    parser.add_argument('--devices', type=str, help='devices', default='0')

    opt = parser.parse_args()
    realtime_detect(opt)