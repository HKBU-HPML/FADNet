from netdef_slim.core.register import register_op
import tensorflow as tf
from .blob import _slice
import numpy as np
import netdef_slim as nd

nothing = None

# ----------------------------------------------------------------------
def _threshold(tensor, thresh):
    condition = tf.less(tensor, thresh)
    return tf.where(condition, _const_like(tensor, 0.0), _const_like(tensor, 1.0))

register_op('threshold', _threshold)

# ----------------------------------------------------------------------
_scale_conv_n=0
def _scale(tensor, factor):
    '''
    Factor can be a scalar or a list. In case of a list
    each channel is scaled by a different factor.
    '''
    global _scale_conv_n
    if type(factor) is list or type(factor) is tuple:
        _scale_conv_n += 1
        kernel = tf.constant(factor)
        kernel = tf.reshape(kernel, (1,1,len(factor),1))
        return tf.nn.depthwise_conv2d(tensor,
                            filter=kernel,
                            strides=[1,1,1,1],
                            padding='VALID',
                            data_format='NCHW')

    else:
        return tf.multiply(tensor, factor)

register_op('scale', _scale)

# ----------------------------------------------------------------------
def _zeros(num, channels, height, width, include_phase=None):
    return tf.zeros((num, channels, height, width))

register_op('zeros', _zeros)

# ----------------------------------------------------------------------
def _zeros_like(other, include_phase=None):
    return tf.zeros_like(other)

register_op('zeros_like', _zeros_like)

# ----------------------------------------------------------------------
def _ones(num, channels, height, width, include_phase=None):
    return tf.ones((num, channels, height, width))

register_op('ones', _ones)

# ----------------------------------------------------------------------
def _ones_like(other, include_phase=None):
    return tf.ones_like(other)

register_op('ones_like', _ones_like)

# ----------------------------------------------------------------------
def _constant(num, channels, height, width, value):
    return tf.fill((num, channels, height, width), value)

register_op('constant', _constant)

# ----------------------------------------------------------------------
def _const_like(other, value):
    return tf.fill(other.get_shape(), value)

register_op('const_like', _const_like)

# ----------------------------------------------------------------------
def _abs(A):
    return tf.abs(A)

register_op('abs', _abs)

# ----------------------------------------------------------------------
def _add(A, B, coeffA=None, coeffB=None):
    if coeffA is None:
        coeffA = tf.constant(1, dtype=A.dtype)
    if coeffB is None:
        coeffB = tf.constant(1, dtype=B.dtype)
    #return tf.add(tf.multiply(tf.to_float(A), coeffA), tf.multiply(tf.to_float(B), coeffB))
    return tf.add(tf.multiply(A, coeffA), tf.multiply(B, coeffB))

register_op('add', _add)

# ----------------------------------------------------------------------
def _sub(A, B):
    return tf.subtract(A, B)

register_op('sub', _sub)

# ----------------------------------------------------------------------
def _mul(A, B):
    return tf.multiply(A, B)

register_op('mul', _mul)

# ----------------------------------------------------------------------
def _const_mul(coeff, A):
    return tf.scalar_mul(coeff, A)

register_op('const_mul', _const_mul)

# ----------------------------------------------------------------------
def _channel_norm(blob):
    out = tf.sqrt(tf.reduce_sum(tf.square(blob), axis=1))
    return tf.expand_dims(out, axis = 1)


register_op('channel_norm', _channel_norm)
# ----------------------------------------------------------------------
def _sqrt(x):
    return tf.pow(x, 0.5)

register_op('sqrt', _sqrt)

# ----------------------------------------------------------------------
def _sqr(x):
    return tf.pow(x, 2)

register_op('sqr', _sqr)

# ----------------------------------------------------------------------
def _exp(X, scale=1.0):
    return tf.exp(_mul(X, scale))

register_op('exp', _exp)

# ----------------------------------------------------------------------
def _log(x):
    return tf.log(x)

register_op('log', _log)

# ----------------------------------------------------------------------
def _inv(x):
    return tf.pow(x, -1)

register_op('inv', _inv)

# ----------------------------------------------------------------------
def _flip_sign(x):
    return _scale(x, -1)

register_op('flip_sign', _flip_sign)

# ----------------------------------------------------------------------
def _spatial_epe(tensor1, tensor2):
    diff = tensor1 - tensor2
    return _channel_norm(diff)

register_op('spatial_epe', _spatial_epe)

# ----------------------------------------------------------------------
def _softmax(X, axis=1):
    return tf.nn.softmax(X, dim=axis)

register_op('softmax', _softmax)

# ----------------------------------------------------------------------
def _sigmoid(X):
    return tf.sigmoid(X)

register_op('sigmoid', _sigmoid)

# ----------------------------------------------------------------------
def _add_eps(X, eps=1e-2 / 2.0):
    return _add(X, _const_like(X, eps))

register_op('add_eps', _add_eps)

# ----------------------------------------------------------------------
def _derivative(blob, direction, order=1, extent=2):
    if order!=1:
        raise NotImplementedError
    if extent!=2:
        raise NotImplementedError
    if direction == 'x':
        blob = tf.pad(blob, [[0, 0], [0, 0], [0, 0], [0, 1]])
        return tf.abs(blob[:, :, :, 1:] - blob[:, :, :, :-1])
    elif direction == 'y':
        blob = tf.pad(blob, [[0, 0], [0, 0], [0, 1], [0, 0]])
        return tf.abs(blob[:, :, 1:, :] - blob[:, :, :-1, :])

register_op('derivative', _derivative)

# ----------------------------------------------------------------------
def _arg_max(blob, axis):
    return tf.argmax(blob, axis=axis)

register_op('arg_max', _arg_max)

def _neg_relu(tensor):
    return tf.minimum(tf.constant(0, dtype=tf.float32), tensor)

register_op('neg_relu', _neg_relu)
