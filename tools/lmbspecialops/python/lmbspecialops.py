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
from tensorflow.python.framework import ops
import os
import warnings

if 'LMBSPECIALOPS_LIB' in os.environ:
    _lib_path = os.environ['LMBSPECIALOPS_LIB']
else:  # try to find the lib in the build directory relative to this file
    _lib_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..', 'build', 'lib', 'lmbspecialops.so'))
if not os.path.isfile(_lib_path):
    raise ValueError(
        'Cannot find lmbspecialops.so . Set the environment variable LMBSPECIALOPS_LIB to the path to lmbspecialops.so file')
lmbspecialopslib = tf.load_op_library(_lib_path)
print('Using {0}'.format(_lib_path), flush=True)


# create alias for each op
depth_to_flow = lmbspecialopslib.depth_to_flow
depth_to_normals = lmbspecialopslib.depth_to_normals
flow_to_depth2 = lmbspecialopslib.flow_to_depth2
leaky_relu = lmbspecialopslib.leaky_relu
median3x3_downsample = lmbspecialopslib.median3x3_downsample
replace_nonfinite = lmbspecialopslib.replace_nonfinite
scale_invariant_gradient = lmbspecialopslib.scale_invariant_gradient
warp2d = lmbspecialopslib.warp2d

flow_warp = lmbspecialopslib.flow_warp
correlation = lmbspecialopslib.correlation
correlation_1d = lmbspecialopslib.correlation1d
flow_out_of_frame = lmbspecialopslib.flow_out_of_frame
resample = lmbspecialopslib.resample

# Remove this?
# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
#def resample(input, size, antialias=False):
#    """
#    Resamples the given input tensor to a given size.
#    :param input: dtype input tensor [batch, height, width, channels]
#    :param size: new size (height, width)
#    :param antialias: perform antialiasing iff downsampling
#    :return: resized tensor
#    """
#    return lmbspecialopslib.resample(input, size, antialias=antialias)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def decode_ppm(encoded_ppm, dtype=tf.uint8):
    """
    Decodes the input scalar as ppm file
    :param input: scalar with encoded ppm file
    :param dtype: data type of the result
    :return: (dtype) tensor [height, width, 3]
    """
    return lmbspecialopslib.decode_ppm(encoded_ppm, dtype=dtype)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def decode_pfm(encoded_pfm, channels=0):
    """
    Decode a PFM-encoded image to a uint8 or uint16 tensor.
    :param encoded_pfm: scalar with encoded ppm file
    :param channels: number of channels (0: auto, 1 or 3)
    :return: float32 tensor [height, width, channels], scale scalar (absolute)
    """
    return lmbspecialopslib.decode_pfm(encoded_pfm, channels=channels)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def decode_flo(encoded_flo):
    """
    Decodes the input scalar as flo file
    :param input: scalar with encoded flo file
    :return: float32 tensor [height, width, 2]
    """
    return lmbspecialopslib.decode_flo(encoded_flo)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def decode_webp(encoded_webp, channels=0):
    """
    Decode a WEBP-encoded image to a uint8 tensor.
    :param encoded_webp: scalar with encoded webp file
    :param channels: number of channels (0: auto, 3 or 4)
    :return: uint8 tensor [height, width, channels]
    """
    return lmbspecialopslib.decode_webp(encoded_webp, channels=channels)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def encode_flo(input):
    """
    Encodes the input tensor as flo file
    :param input: float32 tensor [height, width, 2]
    :return: string scalar
    """
    return lmbspecialopslib.encode_flo(input)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def encode_pfm(input, scale=-1.0):
    """
    Encodes the input tensor as flo file
    :param input: float32 tensor [height, width, 2]
    :param scale: float32 scalar containing the scale. Negative for little endian encoding.
    :return: string scalar
    """
    return lmbspecialopslib.encode_pfm(input, scale=scale)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def encode_webp(input,

                lossless=False,

                lossless_level=6,

                preset='default',
                preset_quality=75,

                quality=-1,
                method=-1,
                image_hint='unspecified',

                target_size=-1,
                target_PSNR=0,
                segments=-1,
                sns_strength=-1,
                filter_strength=-1,
                filter_sharpness=-1,
                filter_type=-1,
                autofilter=-1,
                alpha_compression=-1,
                alpha_filtering=-1,
                alpha_quality=-1,
                pass_=-1,

                show_compressed=False,
                preprocessing=-1,
                partitions=0,
                partition_limit=-1,
                emulate_jpeg_size=False,
                thread_level=False,
                low_memory=False,

                near_lossless=100,
                exact=False):
    """
    Encodes the input tensor as flo file

    :param image: uint8 tensor with shape `[height, width, {3|4}]`.

    :param contents: 0-D.  The WEBP-encoded data.

    :param lossless: bool, Lossless (or lossy) encoding. Default is false.

    :param lossless_level: int, Activate the lossless compression mode with the desired efficiency level between 0 (fastest,
      lowest compression) and 9 (slower, best compression). A good default level is '6', providing a fair tradeoff between
      compression speed and final compressed size.

    :param preset: int, used to initialize the encoder settings:
      0: Default preset.
      1: Picture: Digital picture, like portrait, inner shot
      2: Photo: Outdoor photograph, with natural lighting
      3: Drawing: Hand or line drawing, with high-contrast details
      4: Icon. Small-sized colorful images
      5: Text: Text-like
    :param preset_quality: float, between 0 (smallest file) and 100 (biggest), used to initialize the encoder settings.

    Encoder configuration:
    :param quality: float, between 0 (smallest file) and 100 (biggest)
    :param method: int, quality/speed trade-off (0=fast, 6=slower-better)

    :param image_hint: int, Hint for image type (lossless only for now):
      0: default preset.
      1: digital picture, like portrait, inner shot
      2: outdoor photograph, with natural lighting
      3: Discrete tone image (graph, map-tile etc).

    Parameters related to lossy compression only:
    :param target_size: int, if non-zero, set the desired target size in bytes. Takes precedence over the 'compression' parameter.
    :param target_PSNR: float, if non-zero, specifies the minimal distortion to try to achieve. Takes precedence over target_size.
    :param segments: int, maximum number of segments to use, in [1..4]
    :param sns_strength: int, Spatial Noise Shaping. 0=off, 100=maximum.
    :param filter_strength: int, range: [0 = off .. 100 = strongest]
    :param filter_sharpness: int, range: [0 = off .. 7 = least sharp]
    :param filter_type: int, filtering type: 0 = simple, 1 = strong (only used if filter_strength > 0 or autofilter > 0)
    :param autofilter: int, Auto adjust filter's strength [0 = off, 1 = on]
    :param alpha_compression: int, Algorithm for encoding the alpha plane (0 = none, 1 = compressed with WebP lossless). Default is 1.
    :param alpha_filtering: int, Predictive filtering method for alpha plane. 0: none, 1: fast, 2: best. Default if 1.
    :param alpha_quality: int, Between 0 (smallest size) and 100 (lossless). Default is 100.
    :param pass_: int, number of entropy-analysis passes (in [1..10]).

    :param show_compressed: bool, if set, export the compressed picture back. In-loop filtering is not applied.
    :param preprocessing: int, preprocessing filter:
      0=none, 1=segment-smooth, 2=pseudo-random dithering
    :param partitions: int, log2(number of token partitions) in [0..3]. Default is set to 0 for easier progressive decoding.
    :param partition_limit: int, quality degradation allowed to fit the 512k limit on prediction modes coding (0: no degradation,
      100: maximum possible degradation).
    :param emulate_jpeg_size: bool, If true, compression parameters will be remapped to better match the expected output size from
      JPEG compression. Generally, the output size will be similar but the degradation will be lower.
    :param thread_level: bool, If set, try and use multi-threaded encoding.
    :param low_memory: bool, If set, reduce memory usage (but increase CPU use).

    :param near_lossless: int, Near lossless encoding [0 = max loss .. 100 = off (default)].
    :param exact: bool, if set, preserve the exact RGB values under transparent area. Otherwise, discard this invisible RGB
      information for better compression. The default value is false.

    :return: string scalar
    """
    return lmbspecialopslib.encode_webp(input, lossless=lossless, lossless_level=lossless_level, preset=preset,
                                        preset_quality=preset_quality, quality=quality, method=method,
                                        image_hint=image_hint, target_size=target_size,
                                        target_PSNR=target_PSNR, segments=segments,
                                        sns_strength=sns_strength, filter_strength=filter_strength,
                                        filter_sharpness=filter_sharpness, filter_type=filter_type,
                                        autofilter=autofilter, alpha_compression=alpha_compression,
                                        alpha_filtering=alpha_filtering, alpha_quality=alpha_quality,
                                        pass_=pass_, show_compressed=show_compressed, preprocessing=preprocessing,
                                        partitions=partitions, partition_limit=partition_limit,
                                        emulate_jpeg_size=emulate_jpeg_size, thread_level=thread_level,
                                        low_memory=low_memory, near_lossless=near_lossless,
                                        exact=exact)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def encode_lz4(input):
    """
    Encodes the input tensor as lz4
    :param input: tensor
    :return: scalar tf.string
    """
    return lmbspecialopslib.encode_lz4(input)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def decode_lz4(input, expected_shape, dtype):
    """
    Decodes the given input as tensor
    :param input: scalar string
    :param expected_shape: Output shape
    :param dtype: Output datatype
    :return: dtyped tensor, expected_shape shaped
    """
    return lmbspecialopslib.decode_lz4(input, expected_shape=expected_shape, dtype=dtype)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def encode_lz4_raw(input):
    """
    Encodes the given input string as lz4 encoded string
    :param input: Some binary string
    :return: lz4 encoded input scalar string
    """
    return lmbspecialopslib.encode_lz4_raw(input)


# Author: Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
def decode_lz4_raw(input, expected_size):
    """
    Decodes the given lz4 encoded input string as string
    :param input: the lz4 encoded scalar string
    :param expected_size: Maximum size of the output (the actual output might be smaller)
    :return: decoded scalar string
    """
    return lmbspecialopslib.decode_lz4_raw(input, expected_size=expected_size)


# wrap deprecated functions
def flow_to_depth(flow, intrinsics, rotation, translation, rotation_format=None, inverse_depth=None, normalized_flow=None, name=None, nowarning=False):
    if not nowarning:
        warnings.warn("flow_to_depth has incorrect behaviour but is kept for compatibility. Please use flow_to_depth2", DeprecationWarning, stacklevel=2)
    return lmbspecialopslib.flow_to_depth(
            flow=flow,
            intrinsics=intrinsics,
            rotation=rotation,
            translation=translation,
            rotation_format=rotation_format,
            inverse_depth=inverse_depth,
            normalized_flow=normalized_flow,
            name=name)
flow_to_depth.__doc__ = lmbspecialopslib.flow_to_depth.__doc__


# register gradient ops
@ops.RegisterGradient("ScaleInvariantGradient")
def _scale_invariant_gradient_grad(op, grad):
    return lmbspecialopslib.scale_invariant_gradient_grad(
        gradients=grad,
        input=op.inputs[0],
        deltas=op.get_attr('deltas'),
        weights=op.get_attr('weights'),
        epsilon=op.get_attr('epsilon'))


@ops.RegisterGradient("ReplaceNonfinite")
def _replace_nonfinite_grad(op, grad):
    return lmbspecialopslib.replace_nonfinite_grad(
        gradients=grad,
        input=op.inputs[0])


@ops.RegisterGradient("LeakyRelu")
def _leaky_relu_grad(op, grad):
    return lmbspecialopslib.leaky_relu_grad(
        gradients=grad,
        input=op.inputs[0],
        leak=op.get_attr('leak'))

@ops.RegisterGradient("FlowWarp")
def _flow_warp_grad(op, grad):
    return lmbspecialopslib.flow_warp_grad(
            gradient=grad,
            image=op.inputs[0],
            flow=op.inputs[1])

@ops.RegisterGradient("Correlation")
def _correlation_grad(op, grad):
    return lmbspecialopslib.correlation_grad(
            gradient=grad,
            input1=op.inputs[0],
            input2=op.inputs[1],
            kernel_size=op.get_attr('kernel_size'),
            max_displacement=op.get_attr('max_displacement'),
            stride1=op.get_attr('stride1'),
            stride2=op.get_attr('stride2'),
            pad_size=op.get_attr('pad_size') )

@ops.RegisterGradient("Correlation1D")
def _correlation_1d_grad(op, grad):
    return lmbspecialopslib.correlation1d_grad(
            gradient=grad,
            input1=op.inputs[0],
            input2=op.inputs[1],
            kernel_size=op.get_attr('kernel_size'),
            max_displacement=op.get_attr('max_displacement'),
            stride1=op.get_attr('stride1'),
            stride2=op.get_attr('stride2'),
            pad_size=op.get_attr('pad_size'),
            single_dir=op.get_attr('single_dir')
            )
