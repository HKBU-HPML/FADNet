from netdef_slim.core.base_scope import BaseScope, bottom_scope
from netdef_slim.core.register import register_class
import netdef_slim as nd
import tensorflow as tf

nothing = None

class _Scope(BaseScope):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        if name is not None:
            self._variable_scope = tf.variable_scope(name, reuse=tf.AUTO_REUSE)
            self._variable_scope.__enter__()

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, type, val, tb):
        super().__exit__(type, val, tb)
        if self._name is not None:
            self._variable_scope.__exit__(type, val, tb)

    def name_scope(self): return self._name_scope
    def variable_scope(self): return self._variable_scope
    def lr_config(self): return self._lr_config
    def activation_function(self): return self._activation_function
    def parent(self): return self._parent
    def weight_decay(self): return self._config['weight_decay']
    def learn(self): return self._config['learn']
    def global_step(self):
        if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="global_step") != []:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="global_step")[0]
        else:
            return None


register_class('Scope', _Scope)


bottom_scope = _Scope(None,
                      learn=True,
                      loss_fact=1.0,
                      conv_op=nd.ops.conv,
                      conv_nonlin_op=nd.ops.conv_relu,
                      upconv_op=nd.ops.upconv,
                      upconv_nonlin_op=nd.ops.upconv_relu,
                      weight_decay=0.0)

bottom_scope.__enter__()
