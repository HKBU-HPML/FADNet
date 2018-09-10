from netdef_slim.core.register import register_op
import tensorflow as tf
from lmbspecialops import leaky_relu
import netdef_slim as nd

nothing = None

# ----------------------------------------------------------------------
def _upconv(input, activation=None, **kwargs):


    k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
    b_initializer = tf.zeros_initializer
    k_regularizer = tf.contrib.layers.l2_regularizer(scale=nd.scope.weight_decay()*0.0)

    # for shared params
    dropout = kwargs.pop("dropout", False)
    if dropout: raise NotImplementedError

    kernel_size = kwargs.pop('kernel_size', False)
    num_output = kwargs.pop('num_output', False)
    stride = kwargs.pop('stride', 1)
    pad = kwargs.pop('pad', 0)
    pad = 'same' #### overriding pad, TODO
    name = kwargs.pop('name', None)
    # conv_params = nd.scope.conv_params()

    if not kernel_size:
        raise KeyError('Missing kernel size')
    if not num_output:
        raise KeyError('Missing output size')

    # layer
    # note: input might be a tuple, in which case weights are shared
    if not isinstance(input, tuple):
        deconv_out = tf.layers.conv2d_transpose(inputs=input,
                                                filters=num_output,
                                                kernel_size=kernel_size,
                                                strides=stride,
                                                padding=pad,
                                                data_format='channels_first',
                                                trainable=nd.scope.learn(),
                                                activation=activation,
                                                kernel_initializer = k_initializer,
                                                kernel_regularizer = k_regularizer,
                                                bias_initializer = b_initializer,
                                                use_bias=True,
                                                name=name,
                                                )

        return deconv_out
    else:
        outputs = []
        for i in input:
            outputs.append(tf.layers.conv2d_transpose(inputs=i,
                                                    filters=num_output,
                                                    kernel_size=kernel_size,
                                                    strides=stride,
                                                    padding=pad,
                                                    data_format='channels_first',
                                                    trainable=nd.scope.learn(),
                                                    activation=activation,
                                                    reuse=tf.AUTO_REUSE,
                                                    kernel_initializer = k_initializer,
                                                    kernel_regularizer = k_regularizer,
                                                    bias_initializer = b_initializer,
                                                    use_bias=True,
                                                    name=name,
                                                    ))

        return outputs

register_op('upconv', _upconv)


def _upconv_relu(input, **kwargs):
    return _upconv(input, leaky_relu, **kwargs)
register_op('upconv_relu', _upconv_relu)


def _upconv_elu(input, **kwargs):
    return _upconv(input, tf.nn.elu, **kwargs)
register_op('upconv_elu', _upconv_elu)

def _upconv_bn_relu(input,**kwargs):

    k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
    b_initializer = tf.zeros_initializer
    # We don't regularize the kernels for deconv in caffe
    k_regularizer = tf.contrib.layers.l2_regularizer(scale=nd.scope.weight_decay()) #


    kernel_size = kwargs.pop('kernel_size', False)
    num_output = kwargs.pop('num_output', False)
    stride = kwargs.pop('stride', 1)
    pad = kwargs.pop('pad', 0)
    pad = 'same'
    name = kwargs.pop('name', None)

    if not kernel_size:
        raise KeyError('Missing kernel size')
    if not num_output:
        raise KeyError('Missing output size')

    # layer
    # note: input might be a tuple, in which case weights are shared
    if not isinstance(input, tuple):
        deconv_out = tf.layers.conv2d_transpose(inputs=input,
                                                filters=num_output,
                                                kernel_size=kernel_size,
                                                strides=stride,
                                                padding=pad,
                                                data_format='channels_first',
                                                trainable=nd.scope.learn(),
                                                kernel_initializer = k_initializer,
                                                #kernel_regularizer = k_regularizer,
                                                bias_initializer = b_initializer,
                                                use_bias=True,
                                                name=name,
                                                )
        bn_out = tf.layers.batch_normalization(
            deconv_out,
            axis=1,
            gamma_initializer=tf.constant_initializer(1.0),
            beta_initializer=tf.constant_initializer(0.0),
            scale=True,
            center=True,
            training=bool(nd.phase == 'train'),
            trainable=nd.scope.learn(),
            beta_regularizer = k_regularizer,
            gamma_regularizer = k_regularizer,
            name=name + "_bn"
            )

        return leaky_relu(bn_out)
    else:
        outputs = []
        for i in input:
            deconv_out = tf.layers.conv2d_transpose(inputs=i,
                                                    filters=num_output,
                                                    kernel_size=kernel_size,
                                                    strides=stride,
                                                    padding=pad,
                                                    data_format='channels_first',
                                                    trainable=nd.scope.learn(),
                                                    reuse=tf.AUTO_REUSE,
                                                    kernel_initializer = k_initializer,
                                                    #kernel_regularizer = k_regularizer,
                                                    bias_initializer = b_initializer,
                                                    use_bias=True,
                                                    name=name,
                                                    )

            bn_out = tf.layers.batch_normalization(
                                                    deconv_out,
                                                    axis=1,
                                                    gamma_initializer=tf.constant_initializer(1.0),
                                                    beta_initializer=tf.constant_initializer(0.0),
                                                    scale=True,
                                                    center=True,
                                                    training=bool(nd.phase == 'train'),
                                                    trainable=nd.scope.learn(),
                                                    beta_regularizer = k_regularizer,
                                                    gamma_regularizer = k_regularizer,
                                                    name=name + "_bn",
                                                    reuse=tf.AUTO_REUSE,
                                                    )

        outputs.append(leaky_relu(bn_out))


        return outputs

register_op('upconv_bn_relu', _upconv_bn_relu)
