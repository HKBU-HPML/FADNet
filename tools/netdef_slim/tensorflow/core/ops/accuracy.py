from netdef_slim.core.register import register_op
import netdef_slim as nd
import tensorflow as tf

nothing = None

# ----------------------------------------------------------------------
def _flow_accuracy(pred, gt, name, occ_gt=None):
    name = name.replace('[','-')
    name = name.replace(']','-')
    spatial_epe = nd.ops.spatial_epe(pred, gt)
    loss = tf.reduce_mean(tf.boolean_mask(spatial_epe, tf.is_finite(spatial_epe)) , name=name)
    if occ_gt is not None:
        raise NotImplementedError
    tf.add_to_collection('metric_ops', loss)
    return loss

register_op('flow_accuracy', _flow_accuracy)


def _disp_accuracy(pred, gt, name, occ_gt=None):
    name = name.replace('[','-')
    name = name.replace(']','-')
    spatial_epe = nd.ops.spatial_epe(pred, gt)
    loss = tf.reduce_mean(tf.boolean_mask(spatial_epe, tf.is_finite(spatial_epe)) , name=name)
    if occ_gt is not None:
        raise NotImplementedError
    tf.add_to_collection('metric_ops', loss)
    return loss

register_op('disp_accuracy', _disp_accuracy)
