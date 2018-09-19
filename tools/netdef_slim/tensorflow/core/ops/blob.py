from netdef_slim.core.register import register_op
import tensorflow as tf

nothing = None


# ----------------------------------------------------------------------
def _slice(tensor, slice_points, axis=1):
    '''
    Example: slice(blob, (2,3))
    If the extent is 5, this will return
    blobs of sizes 2, 1, 2.
    '''
    #slices = []
    #previous_slice_point = 0
    #for current_slice_point in slice_points:
        #tmp_slice = tf.slice(tensor,
                             #[previous_slice_point],
                             #[current_slice_point-previous_slice_point])
        #slices.append(tmp_slice)
        #previous_slice_point = current_slice_point


    if not isinstance(slice_points, list):
        slice_points = [slice_points]

    size_splits = []
    last_point = 0
    for p in slice_points:
        size_splits.append(p-last_point)
        last_point = p
    size_splits.append(tensor.get_shape().as_list()[axis]-last_point)

    # print(tensor)
    # print(slice_points)
    # print(size_splits)
    out = tf.split(tensor, size_splits, axis=axis)

    return out

register_op('slice', _slice)


# ----------------------------------------------------------------------
def _blobFromScalar(scalar):
    return tf.Variable(scalar)

register_op('blobFromScalar', _blobFromScalar)


# ----------------------------------------------------------------------
def _concat(*args, **kwargs):
    '''
    The counterpart to sliice. All input blobs will be concatenated.
    '''
    axis = 1
    if 'axis' in kwargs:
        axis = int(kwargs['axis'])
        del kwargs['axis']
    if len(kwargs):
        raise Exception('Cannot handle kwargs %s' % kwargs)

    inputs = []
    for arg in args:
        if isinstance(arg, list) or isinstance(arg, tuple): inputs += arg
        else: inputs.append(arg)

    return tf.concat(inputs, axis)

register_op('concat', _concat)


# ----------------------------------------------------------------------
def _replace_nan(data, value=0.0):
    return tf.where(tf.is_nan(data), tf.ones_like(data)*value, data)

register_op('replace_nan', _replace_nan)

def _blob_copy(data):
    var = tf.get_variable('copy', shape=data.get_shape())
    return var.assign(data)

register_op('blob_copy', _blob_copy)


def _to_nchw(blob):
    return tf.transpose(blob, [0, 3, 1, 2])

register_op('to_nchw', _to_nchw)


def _to_nhwc(blob):
    return tf.transpose(blob, [0, 2, 3, 1])

register_op('to_nhwc', _to_nhwc)



