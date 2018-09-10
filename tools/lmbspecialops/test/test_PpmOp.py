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


def read_ppm_ref(data, dtype=np.uint8):
    """
    Reads data contents as ppm file. Reference implementation using numpy.
    :param file: str
    :return: tuple(data: np.array(h, w, 3), maxval: int)
    """
    import re
    header = data[:2]
    if header != b'P6':
        raise Exception('Not a PPM file')
    ppm_header_re = re.compile(br'^P6\s+(?P<width>\d+)\s+(?P<height>\d+)\s+(?P<maxval>-?\d*\.?\d*)\s')
    ppm_header = ppm_header_re.match(data)
    if not ppm_header:
        raise Exception('Not a PPM file')
    width = int(ppm_header.group('width').decode('utf-8'))
    height = int(ppm_header.group('height').decode('utf-8'))
    maxval = int(ppm_header.group('maxval').decode('utf-8'))

    if maxval > 255:
        assert dtype == np.uint16
        data = np.fromstring(data[ppm_header.end():], dtype='>u2', count=3*width*height)
        return np.reshape(data, newshape=(height, width, 3)), maxval

    assert dtype == np.uint8
    data = np.fromstring(data[ppm_header.end():], dtype='>u1', count=3*width*height)
    return np.reshape(data, newshape=(height, width, 3)), maxval


def write_ppm_ref(input, maxval=255):
    """
    Reads data contents as ppm file. Reference implementation using numpy.
    :param input: np.array (h, w, 3) of type uint8 or uint16
    :param maxval: int, maximum value of the encoded data
    :return: bytearray
    """
    if input.ndim != 3 or input.shape[2] != 3:
        raise Exception('Not PPM encodable')

    if maxval > 255:
        if input.dtype.hasobject or input.dtype.type != np.uint16:
            raise Exception('Not PPM encodable')
    else:
        if input.dtype.hasobject or input.dtype.type != np.uint8:
            raise Exception('Not PPM encodable')
    endian = input.dtype.byteorder
    input_little_endian = (endian == '<' or (endian == '=' and sys.byteorder == 'little'))
    if input_little_endian:
        input = input.byteswap()

    return b'P6\n' + str(input.shape[0]).encode('utf-8') + b' ' + str(input.shape[1]).encode('utf-8') + \
        b'\n' + str(maxval).encode('utf-8') + b'\n' + input.tobytes()


class EncodeDecodePpmTest(tf.test.TestCase):
    def _test_load(self, data, maxval):
        with self.test_session(use_gpu=False, force_gpu=False):
            dtype = tf.uint8 if maxval < 256 else tf.uint16
            encoded_ppm = write_ppm_ref(data, maxval=maxval)
            output_img, output_maxval = ops.decode_ppm(encoded_ppm, dtype=dtype)
            self.assertEqual(maxval, output_maxval.eval())
            self.assertAllEqual(data, output_img.eval())

    def _test_ref(self, data, maxval):
        with self.test_session(use_gpu=False, force_gpu=False):
            dtype = np.uint8 if maxval < 256 else np.uint16
            encoded_ppm = write_ppm_ref(data, maxval=maxval)
            ref_output_data, ref_maxval = read_ppm_ref(encoded_ppm, dtype=dtype)
            self.assertEqual(maxval, ref_maxval)
            self.assertAllEqual(data, ref_output_data)

    @staticmethod
    def _test_data():
        return [
            ((np.random.rand(1, 1, 3) * 256).astype(np.uint8), 255),
            ((np.random.rand(100, 100, 3) * 256).astype(np.uint8), 255),
            ((np.random.rand(1, 1, 3) * 512).astype(np.uint16), 511),
            ((np.random.rand(100, 100, 3) * 512).astype(np.uint16), 511),
        ]

    def test_ref(self):
        for t in self._test_data():
            self._test_ref(t[0], t[1])

    def test_load(self):
        for t in self._test_data():
            self._test_load(t[0], t[1])


if __name__ == '__main__':
    tf.test.main()




