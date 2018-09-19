#!/usr/bin/env python
import cv2
import argparse
import time

DATAPATH='/media/sf_Shared_Data/gpuhomedataset/dispnet/custom'

def get_cap_info(cap1, cap2):
    width = cap2.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap2.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    #print('width: ', width, ', height: ', height)
    #jijwidth = 1920
    #height = 1080
    width = 1080
    height = 768

    cap2.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
    cap2.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)

    cap1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)


def binocular_player():
    left_cap = cv2.VideoCapture(0)
    right_cap = cv2.VideoCapture(1)
    nframes=120
    t = time.time()
    counter = 0
    get_cap_info(left_cap, right_cap)
    caped = 0
    while(left_cap.isOpened()):
        ret1, frame1 = left_cap.read()
        ret2, frame2 = right_cap.read()
        counter+=1
        if ret2:
            cv2.imshow('left', frame1)
            cv2.imshow('right', frame2)
        if counter % nframes == 0:
            cv2.imwrite('%s/left/left_%d.png'%(DATAPATH, caped), frame1)
            cv2.imwrite('%s/right/right_%d.png'%(DATAPATH, caped), frame2)
            caped += 1
            print('image shape: ', frame1.shape, ', fps: ', nframes/(time.time()-t))
            t = time.time()
            counter = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binocular webcam player')
    
    args = parser.parse_args()
    binocular_player()
