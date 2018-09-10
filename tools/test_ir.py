from skimage import io
import numpy as np

left_name = 
right_name = 
ir_left_name = 
ir_right_name = 

img_left = io.imread(left_name)[:, :, 0:3]
img_left = io.imread(right_name)[:, :, 0:3]

ir_left = io.imread(ir_left_name)[:, :, 0]
ir_right = io.imread(ir_right_name)[:, :, 0]
with_ir_left = np.zeros(shape=(img_left.shape[0], img_left.shape[1], 4), dtype=ir_left.dtype)
with_ir_right = np.zeros(shape=(img_left.shape[0], img_left.shape[1], 4), dtype=ir_left.dtype)
with_ir_left[:,:,0:3] = img_left[:,:,0:3]
with_ir_left[:,:,3] = ir_left
with_ir_right[:,:,0:3] = img_right[:,:,0:3]
with_ir_right[:,:,3] = ir_right
img_right = with_ir_right
img_left = with_ir_left

