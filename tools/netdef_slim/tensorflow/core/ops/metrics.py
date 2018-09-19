from netdef_slim.core.register import register_op
import netdef_slim as nd
import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np

nothing = None

def _py_fmeasure(pred, gt):
    valid_gt   = gt[np.logical_not(np.isnan(gt))]
    valid_pred = pred[np.logical_not(np.isnan(gt))]
    return f1_score(valid_gt.flatten()>0, valid_pred.flatten()>0)

def _f_measure(pred, gt, name):
    name = name.replace('[','-').replace(']','-')
    pred = tf.to_float(pred)
    gt = tf.to_float(gt)
    fm = tf.py_func(_py_fmeasure, [pred, gt], tf.double, name = name, stateful=False)
    tf.add_to_collection('metric_ops', fm)
    return fm

register_op('f_measure', _f_measure)
