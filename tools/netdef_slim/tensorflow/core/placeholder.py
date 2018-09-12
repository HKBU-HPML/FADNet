from netdef_slim.core.register import register_function
import tensorflow as tf
nothing = None

def _placeholder(name, shape, dtype=tf.float32):
    ph = tf.placeholder(dtype,
                        shape=shape,
                        name=name
                        )
    tf.add_to_collection('placeholders', ph)
    return ph
register_function('placeholder', _placeholder)
