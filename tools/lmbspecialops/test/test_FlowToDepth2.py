#
#  lmbspecialops - a collection of tensorflow ops
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import tensorflow as tf
import numpy as np
import sys
print(sys.path)
sys.path.insert(0,'../python')
import lmbspecialops as ops
from helper import *
np.set_printoptions(linewidth=160)

USE_GPUS = sorted(set((False, tf.test.is_gpu_available())))
TYPES = (np.float32, np.float64)

#USE_GPUS = [False]
#TYPES = (np.float32, )


class FlowToDepth2Test(tf.test.TestCase):


    def _test_random_data(self, dtype, inverse_depth, normalize_flow):

        # random depth map and camera pose
        depth = np.random.uniform(5,10, (1,1,6,12)).astype(dtype)
        if inverse_depth:
            depth = 1/depth
        rotation = np.random.uniform(0.0,0.05, (1,3)).astype(dtype)
        translation = (np.array([[1,0,0]]) + np.random.uniform(-0.2,0.2, (1,3))).astype(dtype)
        intrinsics = np.array([[1,1,0.5,0.5]]).astype(dtype)


        flow = ops.depth_to_flow(
            depth=depth, 
            intrinsics=intrinsics,
            rotation=rotation, 
            translation=translation, 
            inverse_depth=inverse_depth, 
            normalize_flow=normalize_flow,).eval()

        # rotation = angleaxis_to_rotation_matrix(rotation[0])[np.newaxis,:,:]
        rotation = angleaxis_to_quaternion(rotation[0])[np.newaxis,:]

        computed_depth = ops.flow_to_depth2(
            flow=flow, 
            intrinsics=intrinsics,
            rotation=rotation, 
            translation=translation, 
            inverse_depth=inverse_depth, 
            normalized_flow=normalize_flow,
            rotation_format='quaternion').eval()

        print('depth\n',depth)
        print('computed_depth\n',computed_depth)
        self.assertAllClose(depth, computed_depth, rtol=1e-4, atol=1e-4)

    def test_random_data(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
                    for inverse_depth in (False, True):
                        for normalize_flow in (False, True):
                            self._test_random_data(dtype, inverse_depth, normalize_flow)



    def _test_rotation_formats(self, dtype, inverse_depth, normalize_flow):

        # random depth map and camera pose
        depth = np.random.uniform(5,10, (1,1,6,12)).astype(dtype)
        if inverse_depth:
            depth = 1/depth
        rotation = np.random.uniform(0.0,0.05, (1,3)).astype(dtype)
        translation = (np.array([[1,0,0]]) + np.random.uniform(-0.2,0.2, (1,3))).astype(dtype)
        intrinsics = np.array([[1,1,0.5,0.5]]).astype(dtype)

        flow = ops.depth_to_flow(
            depth=depth, 
            intrinsics=intrinsics,
            rotation=rotation, 
            translation=translation, 
            inverse_depth=inverse_depth, 
            normalize_flow=normalize_flow,).eval()

        rotation_aa = rotation
        rotation_R = angleaxis_to_rotation_matrix(rotation[0])[np.newaxis,:,:]
        rotation_q = angleaxis_to_quaternion(rotation[0])[np.newaxis,:]

        computed_depth_aa = ops.flow_to_depth2(
            flow=flow, 
            intrinsics=intrinsics,
            rotation=rotation_aa, 
            translation=translation, 
            inverse_depth=inverse_depth, 
            normalized_flow=normalize_flow,
            rotation_format='angleaxis3').eval()
        
        computed_depth_R = ops.flow_to_depth2(
            flow=flow, 
            intrinsics=intrinsics,
            rotation=rotation_R, 
            translation=translation, 
            inverse_depth=inverse_depth, 
            normalized_flow=normalize_flow,
            rotation_format='matrix').eval()

        computed_depth_q = ops.flow_to_depth2(
            flow=flow, 
            intrinsics=intrinsics,
            rotation=rotation_q, 
            translation=translation, 
            inverse_depth=inverse_depth, 
            normalized_flow=normalize_flow,
            rotation_format='quaternion').eval()

        self.assertAllClose(computed_depth_aa, computed_depth_R, rtol=1e-4, atol=1e-4)
        self.assertAllClose(depth, computed_depth_q, rtol=1e-4, atol=1e-4)

    def test_rotation_formats(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
                    for inverse_depth in (False, True):
                        for normalize_flow in (False, True):
                            self._test_rotation_formats(dtype, inverse_depth, normalize_flow)


    def test_shape_no_batch_dimension(self):
        dtype = np.float32

        flow = np.zeros((2,6,12)).astype(dtype)
        rotation = np.zeros((3,)).astype(dtype)
        translation = np.array([1,0,0]).astype(dtype)
        intrinsics = np.array([1,1,0.5,0.5]).astype(dtype)

        depth = np.zeros((1,1,6,12)).astype(dtype)
        computed_depth = ops.flow_to_depth2(
            flow=flow, 
            intrinsics=intrinsics,
            rotation=rotation, 
            translation=translation,)
        self.assertShapeEqual(depth, computed_depth)


    def test_shape_batch(self):
        dtype = np.float32

        batch = 7

        flow = np.zeros((batch,2,6,12)).astype(dtype)
        rotation = np.zeros((batch,3)).astype(dtype)
        translation = np.zeros((batch,3)).astype(dtype)
        intrinsics = np.zeros((batch,4)).astype(dtype)

        depth = np.zeros((batch,1,6,12)).astype(dtype)
        computed_depth = ops.flow_to_depth2(
            flow=flow, 
            intrinsics=intrinsics,
            rotation=rotation, 
            translation=translation,)
        self.assertShapeEqual(depth, computed_depth)


    def test_shape_batch_mismatch(self):
        dtype = np.float32

        batch = np.array([7,7,7,5],dtype=np.int32)

        for i in range(4):
            batch = np.roll(batch,1)
            print(batch)
            flow = np.zeros((batch[0],2,6,12)).astype(dtype)
            rotation = np.zeros((batch[1],3)).astype(dtype)
            translation = np.zeros((batch[2],3)).astype(dtype)
            intrinsics = np.zeros((batch[3],4)).astype(dtype)

            with self.assertRaises(ValueError) as cm:
                computed_depth = ops.flow_to_depth2(
                    flow=flow, 
                    intrinsics=intrinsics,
                    rotation=rotation, 
                    translation=translation,)
            self.assertStartsWith(str(cm.exception), 'Dimensions must be equal')



if __name__ == '__main__':
    tf.test.main()


