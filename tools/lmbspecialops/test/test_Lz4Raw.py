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
import os
import base64

sys.path.insert(0, '../python')
import lmbspecialops as ops

class EncodeDecodeLz4RawTest(tf.test.TestCase):
    def _test_save_load(self, data):
        with self.test_session(use_gpu=False, force_gpu=False):
            input_data = tf.constant(data)
            encoded_lz4 = ops.encode_lz4_raw(input_data)
            decoded_lz4 = ops.decode_lz4_raw(encoded_lz4, expected_size=len(data))
            self.assertEqual(input_data.eval(), decoded_lz4.eval())
            print("Compression ratio for", len(data), ":", len(encoded_lz4.eval())/len(decoded_lz4.eval()))

    def test_save_load(self):
        for i in (1, 10, 32, 1024, 1024*1024):
            self._test_save_load(os.urandom(i))


if __name__ == '__main__':
    tf.test.main()
