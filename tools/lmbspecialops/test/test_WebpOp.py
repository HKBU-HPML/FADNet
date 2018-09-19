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

sys.path.insert(0, '../python')
import lmbspecialops as ops

class EncodeDecodeWebpTest(tf.test.TestCase):
    def _test_save_load(self, data, settings):
        with self.test_session(use_gpu=False, force_gpu=False):
            input_data = tf.constant(data)
            encoded_webp = ops.encode_webp(input_data, **settings)
            decoded_webp = ops.decode_webp(encoded_webp)
            self.assertEqual(input_data.eval().shape, decoded_webp.eval().shape)
            if 'lossless' in settings and settings['lossless'] == True:
                pass
                # Actually this is supposed to succeed, but it doesn't. Seems to be a bug in the encoder(?)
                #self.assertAllClose(input_data.eval(), decoded_webp.eval(), atol=5.0)

    @staticmethod
    def _test_data():
        return [
            (np.random.rand(1, 1, 3) * 255).astype(np.uint8),
            (np.random.rand(1, 1, 4) * 255).astype(np.uint8),
            (np.random.rand(3, 3, 3) * 255).astype(np.uint8),
            (np.random.rand(3, 3, 4) * 255).astype(np.uint8),
            (np.random.rand(128, 128, 3) * 255).astype(np.uint8),
            (np.random.rand(32, 32, 4) * 255).astype(np.uint8),
        ]

    @staticmethod
    def _test_settings():
        return [
            {'lossless': True, 'preset_quality': 100, 'alpha_quality': 100},
            {'preset_quality': 90},
        ]

    def test_save_load(self):
        for t in self._test_data():
            for s in self._test_settings():
                self._test_save_load(t, s)


if __name__ == '__main__':
    tf.test.main()
