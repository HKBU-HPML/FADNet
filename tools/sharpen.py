import cv2
import numpy as np
# Reading in and displaying our image
path='/media/sf_Shared_Data/gpuhomedataset/dispnet/real_release/frames_cleanpass/left'
image = cv2.imread(path+'/img00000.bmp')
cv2.imshow('Original', image)
# Create our shapening kernel, it must equal to one eventually
kernel_sharpening = np.array([[-1,-1,-1], 
        [-1, 9,-1],
        [-1,-1,-1]])
# applying the sharpening kernel to the input image & displaying it.
sharpened = cv2.filter2D(image, -1, kernel_sharpening)
cv2.imshow('Image Sharpening', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
