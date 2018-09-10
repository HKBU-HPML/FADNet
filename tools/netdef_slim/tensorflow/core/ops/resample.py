from netdef_slim.core.register import register_op
import tensorflow as tf
from lmbspecialops import resample

nothing = None

# ----------------------------------------------------------------------
def _crop(image, width, height):
    data = tf.transpose(image, perm=[0,2,3,1])
    data = tf.image.resize_image_with_crop_or_pad(data, [tf.to_int32(height), tf.to_int32(width)])
    return tf.transpose(data, perm=[0,3,1,2])

register_op('crop', _crop)


# ----------------------------------------------------------------------
def _resample(image, width=None, height=None, reference=None, factor=1.0, type='LINEAR', antialias=True):
    
    types = {'LINEAR': tf.image.ResizeMethod.BILINEAR}
    data = image
    # data = tf.transpose(image, perm=[0,2,3,1])

    if reference is not None:
        b, c, h, w = reference.get_shape().as_list()
        # data = tf.image.resize_images(data, [h, w], method=types[type])
        data = resample(data, w, h, antialias, type)
    elif height is not None and width is not None:
        # data = tf.image.resize_images(data, [tf.to_int32(height), tf.to_int32(width)], method=types[type])
        data = resample(data, width, height, antialias, type)
    else:
        raise ValueError


    return tf.stop_gradient(data)
    # return tf.transpose(data, perm=[0,3,1,2])

register_op('resample', _resample)


# ----------------------------------------------------------------------
def _differentiable_resample(image, width=None, height=None, reference=None, factor=1.0, type='LINEAR', antialias=True):
    types = {'LINEAR': tf.image.ResizeMethod.BILINEAR}
    data = tf.transpose(image, perm=[0,2,3,1])

    if reference is not None:
        b, c, h, w = reference.get_shape().as_list()
        data = tf.image.resize_images(data, [h, w], method=types[type])
    elif height is not None and width is not None:
        data = tf.image.resize_images(data, [tf.to_int32(height), tf.to_int32(width)], method=types[type])
    else:
        raise ValueError

    return tf.transpose(data, perm=[0,3,1,2])


register_op('differentiable_resample', _differentiable_resample)


# ----------------------------------------------------------------------
def _random_crop(image, width, height):
    b, h, w, c = reference.get_shape().as_list()
    offset_height = tf.random_uniform((1,), maxval=h-height, dtype=tf.int32)
    offset_width = tf.random_uniform((1,), maxval=w-width, dtype=tf.int32)
    
    data = tf.transpose(image, perm=[0,2,3,1])
    data = tf.image.crop_to_bounding_box(data, offset_height, offset_width, tf.to_int32(height), tf.to_int32(width))
    return tf.transpose(data, perm=[0,3,1,2])

register_op('random_crop', _random_crop)
