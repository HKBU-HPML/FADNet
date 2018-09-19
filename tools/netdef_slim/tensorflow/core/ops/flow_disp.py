from netdef_slim.core.register import register_op
import tensorflow as tf
from lmbspecialops import correlation as correlation
from lmbspecialops import correlation_1d as correlation_1d
from lmbspecialops import flow_out_of_frame
from lmbspecialops import flow_warp as flow_warp
import netdef_slim as nd


nothing = None


def _correlation(input_a, input_b, **kwargs):

    max_displacement = kwargs.pop('max_displacement', False)
    kernel_size = kwargs.pop('kernel_size', False)
    stride1 = kwargs.pop('stride_1', 1)
    stride2 = kwargs.pop('stride_2', 1)
    pad = kwargs.pop('pad', 0)
    name = kwargs.pop('name', 'correlation')

    if not max_displacement:
        raise KeyError('Missing max displacement')
    if not kernel_size:
        raise KeyError('Missing kernel size')

    corr_out = correlation(input1=input_a,
                             input2=input_b,
                             max_displacement=max_displacement,
                             kernel_size=kernel_size,
                             stride1=stride1,
                             stride2=stride2,
                             pad_size=pad)

    corr_out = tf.nn.relu(corr_out)

    return corr_out

register_op('correlation_2d', _correlation)

def _warp(data, flow, nearest=False):
    # TODO nearest?
    return flow_warp(data, flow)

register_op('warp', _warp)

def _correlation_1d(input_a, input_b, **kwargs):

    max_displacement = kwargs.pop('max_displacement', None)
    kernel_size = kwargs.pop('kernel_size', None)
    stride1 = kwargs.pop('stride_1', 1)
    stride2 = kwargs.pop('stride_2', 1)
    pad = kwargs.pop('pad', 0)
    name = kwargs.pop('name', 'correlation_1d')

    if not max_displacement:
        raise KeyError('Missing max displacement')
    if not kernel_size:
        raise KeyError('Missing kernel size')

    corr_out = correlation_1d(input1=input_a,
                              input2=input_b,
                              max_displacement=max_displacement,
                              kernel_size=kernel_size,
                              stride1=stride1,
                              stride2=stride2,
                              pad_size=pad)

    corr_out = tf.nn.relu(corr_out)

    return corr_out

register_op('correlation_1d', _correlation_1d)

# ----------------------------------------------------------------------
def _occ_add_out_of_frame(occ, flow):
    return flow_out_of_frame(flow=flow, occ=occ)

register_op('occ_add_out_of_frame', _occ_add_out_of_frame)



def _disp_to_flow(blob):
    return nd.ops.concat(blob, nd.ops.zeros_like(blob), axis=1)

register_op('disp_to_flow', _disp_to_flow)

