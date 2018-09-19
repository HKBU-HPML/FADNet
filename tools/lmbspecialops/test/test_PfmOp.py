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


def read_pfm_ref(data):
    """
    Reads data contents as pfm file. Reference implementation using numpy.
    :param file: str
    :return: tuple(data: np.array, scale: float)
    """
    import re
    header = data[:2]
    if header == b'Pf':
        channels = 1
    elif header == b'PF':
        channels = 3
    else:
        raise Exception('Not a PFM file')
    pfm_header_re = re.compile(br'^P[fF]\s+(?P<width>\d+)\s+(?P<height>\d+)\s+(?P<scale>-?\d*\.?\d*)\s')
    pfm_header = pfm_header_re.match(data)
    if not pfm_header:
        raise Exception('Not a PFM file')
    width = int(pfm_header.group('width').decode('utf-8'))
    height = int(pfm_header.group('height').decode('utf-8'))
    scale = float(pfm_header.group('scale').decode('utf-8'))

    if scale < 0:
        # little-endian
        dtype = '<f'
        scale = -scale
    else:
        # big-endian
        dtype = '>f'

    data = np.fromstring(data[pfm_header.end():], dtype=dtype, count=channels*width*height)
    return np.reshape(data, newshape=(height, width, channels))[::-1, :, :], scale


def write_pfm_ref(input, scale=-1.0):
    """
    Reads data contents as pfm file. Reference implementation using numpy.
    :param input: np.array (h, w, 1|3)
    :param scale: float, negative for little endian encoding, positive for big endian.
    :return: bytearray
    """
    if input.ndim != 3:
        raise Exception('Not PFM encodable')
    if input.shape[2] == 1:
        header = b'Pf'
    elif input.shape[2] == 3:
        header = b'PF'
    else:
        raise Exception('Not PFM encodable')

    endian = input.dtype.byteorder
    input_little_endian = (endian == '<' or endian == '=' and sys.byteorder == 'little')
    request_little_endian = scale < 0
    if input_little_endian != request_little_endian:
        input = input.byteswap()

    return header + b'\n' + str(input.shape[0]).encode('utf-8') + b' ' + str(input.shape[1]).encode('utf-8') + \
        b'\n' + "{:f}".format(scale).encode('utf-8') + b'\n' + input[::-1,:,:].tobytes()


class EncodeDecodePfmTest(tf.test.TestCase):
    def _test_save_load(self, data, scale=-1.0):
        with self.test_session(use_gpu=False, force_gpu=False):
            input_data = tf.constant(data)
            encoded_pfm = ops.encode_pfm(input_data, scale=scale)
            test_output_img, out_scale = ops.decode_pfm(encoded_pfm)
            self.assertEqual(out_scale.eval(), np.abs(scale))
            self.assertAllEqual(input_data.eval(), test_output_img.eval())

    def _test_save(self, data, scale=-1.0):
        with self.test_session(use_gpu=False, force_gpu=False):
            encoded_pfm = ops.encode_pfm(data, scale=scale)
            ref_output_data, ref_scale = read_pfm_ref(encoded_pfm.eval())
            self.assertAllCloseAccordingToType(np.abs(scale), ref_scale)
            self.assertAllCloseAccordingToType(ref_output_data, data)

    def _test_load(self, data, scale=-1.0):
        with self.test_session(use_gpu=False, force_gpu=False):
            encoded_pfm = write_pfm_ref(data, scale=scale)
            output_img, output_scale = ops.decode_pfm(encoded_pfm)
            self.assertAllCloseAccordingToType(np.abs(scale), output_scale.eval())
            self.assertAllCloseAccordingToType(data, output_img.eval())

    def _test_ref(self, data, scale=-1.0):
        with self.test_session(use_gpu=False, force_gpu=False):
            encoded_pfm = write_pfm_ref(data, scale=scale)
            ref_output_data, ref_scale = read_pfm_ref(encoded_pfm)
            self.assertAllCloseAccordingToType(np.abs(scale), ref_scale)
            self.assertAllCloseAccordingToType(data, ref_output_data)

    @staticmethod
    def _test_data():
        return [
            (np.random.rand(1, 1, 1).astype(np.float32), -1.0),
            (np.random.rand(1, 1, 1).astype(np.float32), 1.0),
            (np.random.rand(1, 1, 1).astype(np.float32), -2.0),
            (np.random.rand(1, 1, 1).astype(np.float32), 2.0),
            (np.random.rand(3, 3, 1).astype(np.float32), -1.0),
            (np.random.rand(3, 3, 1).astype(np.float32) - 0.5, -1.0),
            (np.random.rand(100, 100, 1).astype(np.float32), -1.0),
            (np.random.rand(100, 100, 3).astype(np.float32) - 0.5, -1.0),
            (np.random.rand(100, 100, 3).astype(np.float32), -1.0),
        ]

    def test_ref(self):
        for t in self._test_data():
            self._test_ref(t[0], t[1])

    def test_load(self):
        for t in self._test_data():
            self._test_load(t[0], t[1])

    def test_save(self):
        for t in self._test_data():
            self._test_save(t[0], t[1])

    def test_save_load(self):
        for t in self._test_data():
            self._test_save_load(t[0], t[1])


if __name__ == '__main__':
    tf.test.main()




