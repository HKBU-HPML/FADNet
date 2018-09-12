from netdef_slim.core.register import register_op
import tensorflow as tf

nothing = None


# ----------------------------------------------------------------------
def _image_to_range_01(image, include_phase=None):
    return tf.multiply(image, 1.0/255.0)

register_op('image_to_range_01', _image_to_range_01)


# ----------------------------------------------------------------------
def _image_to_range_255(image):
    return tf.multiply(image, 255.0)

register_op('image_to_range_255', _image_to_range_255)


# ----------------------------------------------------------------------
def _scale_and_subtract_mean(image, mean=0.4):
    return tf.subtract(tf.multiply(image, 1.0/255.0), mean)

register_op('scale_and_subtract_mean', _scale_and_subtract_mean)


# ----------------------------------------------------------------------
def _add_mean_and_scale(image, mean=0.4):
    return tf.multiply(tf.add(image, mean), 255.0)

register_op('add_mean_and_scale', _add_mean_and_scale)
