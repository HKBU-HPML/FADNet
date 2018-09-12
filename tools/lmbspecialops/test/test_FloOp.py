#
#  lmbspecialops - a collection of tensorflow ops
#  Copyright (C) 2017  Albert Ludwigs University of Freiburg, Pattern Recognition and Image Processing, Computer Vision Group
#  Author(s): Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
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
sys.path.insert(0,'../python')
import lmbspecialops as ops


def read_flo_ref(data):
    """
    Reads data contents as flo file. Reference implementation using numpy.
    :param data: bytearray
    :return: np.array (h, w, 2)
    """
    magic = np.fromstring(data[0:4], dtype=np.float32, count=1)
    if 202021.25 != magic:
        raise Exception('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromstring(data[4:8], dtype=np.int32, count=1)[0]
        h = np.fromstring(data[8:12], dtype=np.int32, count=1)[0]
        if len(data) != 12 + 2*w*h*4:
            raise Exception('Invalid flo data size: ' + str(len(data)) + ', expected ' + str(12+8*w*h))
        data = np.fromstring(data[12:], dtype=np.float32, count=2*w*h)
        return np.reshape(data, newshape=(h, w, 2))


def write_flo_ref(input):
    """
    Reads data contents as flo file. Reference implementation using numpy.
    :param input: np.array (h, w, 2)
    :return: bytearray
    """
    import struct
    if input.ndim != 3:
        raise Exception("Cannot encode flo")
    if input.shape[2] != 2:
        raise Exception("Cannot encode flo")
    if input.dtype != np.float32:
        raise Exception("Cannot encode flo")
    return struct.pack("fII", 202021.25, input.shape[1], input.shape[0]) + input.tobytes()


class EncodeDecodeFloTest(tf.test.TestCase):
    def _test_save_load(self, data):
        with self.test_session(use_gpu=False, force_gpu=False):
            test_input_img = tf.constant(data, dtype=tf.float32)
            encoded_flo = ops.encode_flo(test_input_img)
            test_output_img = ops.decode_flo(encoded_flo)
            self.assertAllEqual(test_input_img.eval(), test_output_img.eval())

    def _test_save(self, data):
        with self.test_session(use_gpu=False, force_gpu=False):
            test_input_img = tf.constant(data, dtype=tf.float32)
            encoded_flo = ops.encode_flo(test_input_img)
            ref_output_data = read_flo_ref(encoded_flo.eval())
            self.assertAllCloseAccordingToType(ref_output_data, data)

    def _test_load(self, data):
        with self.test_session(use_gpu=False, force_gpu=False):
            encoded_flo = write_flo_ref(data)
            output_img = ops.decode_flo(encoded_flo)
            self.assertAllCloseAccordingToType(data, output_img.eval())

    def _test_ref(self, data):
        with self.test_session(use_gpu=False, force_gpu=False):
            encoded_flo = write_flo_ref(data)
            ref_output_data = read_flo_ref(encoded_flo)
            self.assertAllCloseAccordingToType(data, ref_output_data)

    @staticmethod
    def _test_data():
        return [
            np.random.rand(1, 1, 2).astype(np.float32),
            np.random.rand(3, 3, 2).astype(np.float32),
            np.random.rand(3, 3, 2).astype(np.float32) - 0.5,
            np.random.rand(100, 100, 2).astype(np.float32),
            np.random.rand(100, 100, 2).astype(np.float32) - 0.5,
        ]


    def test_ref(self):
        for t in self._test_data():
            self._test_ref(t)

    def test_load(self):
        for t in self._test_data():
            self._test_load(t)

    def test_save(self):
        for t in self._test_data():
            self._test_save(t)

    def test_save_load(self):
        for t in self._test_data():
            self._test_save_load(t)


if __name__ == '__main__':
    tf.test.main()




