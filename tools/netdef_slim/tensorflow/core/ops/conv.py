from netdef_slim.core.register import register_op
import tensorflow as tf
from lmbspecialops import leaky_relu
import netdef_slim as nd

nothing = None


def pad_input(input, pad):  # not to be registered
    padded = tf.pad(input, [[0,0],[0,0],[pad,pad],[pad,pad]])
    return padded

# ----------------------------------------------------------------------

def _conv(input, activation=None, **kwargs):

    k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
    k_regularizer = tf.contrib.layers.l2_regularizer(scale=nd.scope.weight_decay())
    b_initializer = tf.zeros_initializer

    # for shared params
    dropout = kwargs.pop("dropout", False)
    if dropout: raise NotImplementedError

    kernel_size = kwargs.pop('kernel_size', False)
    num_output = kwargs.pop('num_output', False)
    stride = kwargs.pop('stride', 1)
    pad = kwargs.pop('pad', 0)
    name = kwargs.pop('name', 'conv_no_name')

    if not kernel_size:
        raise KeyError('Missing kernel size')
    if not num_output:
        raise KeyError('Missing output size')

    # layer
    # note: input might be a tuple, in which case weights are shared
    if not isinstance(input, tuple):
        conv_out = tf.layers.conv2d(pad_input(input, pad),
                                num_output,
                                kernel_size,
                                strides=stride,
                                data_format='channels_first',
                                trainable=nd.scope.learn(),
                                activation=activation,
                                kernel_regularizer = k_regularizer,
                                kernel_initializer = k_initializer,
                                bias_initializer = b_initializer,
                                name=name)
        return conv_out
    else:
        outputs = []
        for i in input:
            outputs.append(tf.layers.conv2d(pad_input(i, pad),
                                num_output,
                                kernel_size,
                                strides=stride,
                                data_format='channels_first',
                                trainable=nd.scope.learn(),
                                reuse=tf.AUTO_REUSE,
                                activation=activation,
                                kernel_regularizer = k_regularizer,
                                kernel_initializer = k_initializer,
                                bias_initializer = b_initializer,
                                name=name,
                                ))

        return outputs

register_op('conv', _conv)


def _conv_relu(input, **kwargs):
    return _conv(input, activation=leaky_relu, **kwargs)
register_op('conv_relu', _conv_relu)

def _conv_elu(input, **kwargs):
    return _conv(input, activation=tf.nn.elu, **kwargs)
register_op('conv_elu', _conv_elu)



def _conv_bn_relu(input, **kwargs):

    k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
    k_regularizer = tf.contrib.layers.l2_regularizer(scale=nd.scope.weight_decay())
    b_initializer = tf.zeros_initializer

    # for shared params
    dropout = kwargs.pop("dropout", False)
    if dropout: raise NotImplementedError

    kernel_size = kwargs.pop('kernel_size', False)
    num_output = kwargs.pop('num_output', False)
    stride = kwargs.pop('stride', 1)
    pad = kwargs.pop('pad', 0)
    name = kwargs.pop('name', 'conv_no_name')

    if not kernel_size:
        raise KeyError('Missing kernel size')
    if not num_output:
        raise KeyError('Missing output size')

    # layer
    # note: input might be a tuple, in which case weights are shared
    if not isinstance(input, tuple):
        conv_out = tf.layers.conv2d(pad_input(input, pad),
                                num_output,
                                kernel_size,
                                strides=stride,
                                data_format='channels_first',
                                trainable=nd.scope.learn(),
                                kernel_regularizer = k_regularizer,
                                kernel_initializer = k_initializer,
                                bias_initializer = b_initializer,
                                name=name)
        bn_out = tf.layers.batch_normalization(
                conv_out,
                axis=1,
                gamma_initializer=tf.constant_initializer(1.0),
                beta_initializer=tf.constant_initializer(0.0),
                scale=True,
                center=True,
                training=bool(nd.phase == 'train'),
                trainable= nd.scope.learn(),
                beta_regularizer = k_regularizer,
                gamma_regularizer = k_regularizer,
                name = name + '_bn'
                )
        return leaky_relu(bn_out)
    else:
        outputs = []
        for i in input:
            conv_out= tf.layers.conv2d(pad_input(i, pad),
                                num_output,
                                kernel_size,
                                strides=stride,
                                data_format='channels_first',
                                trainable=nd.scope.learn(),
                                reuse=tf.AUTO_REUSE,
                                kernel_regularizer = k_regularizer,
                                kernel_initializer = k_initializer,
                                bias_initializer = b_initializer,
                                name=name,
                                )
            bn_out = tf.layers.batch_normalization(
                conv_out,
                axis=1,
                gamma_initializer=tf.constant_initializer(1.0),
                beta_initializer=tf.constant_initializer(0.0),
                scale=True,
                center=True,
                training=bool(nd.phase == 'train'),
                trainable= nd.scope.learn(),
                beta_regularizer = k_regularizer,
                gamma_regularizer = k_regularizer,
                name = name + '_bn',
                reuse=tf.AUTO_REUSE,
                )
            outputs.append(leaky_relu(bn_out))

        return outputs

register_op('conv_bn_relu', _conv_bn_relu)
