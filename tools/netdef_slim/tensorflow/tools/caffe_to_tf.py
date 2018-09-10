import tensorflow as tf
from math import ceil
import numpy as np
import tb
import os
import h5py

parameter_names_list=None

def make_feed_dict(data, images_placeholder, edge_features_placeholder):
    img0, img1 = resample_input(data)
    input_data = tf.concat((img0, img1), axis=1)
    
    return {images_placeholder: input_data, edge_features_placeholder: img0}

def load_variables(sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(path))

def latest_checkpoint_name(model_folder):
    checkpoints_dir = os.path.join(model_folder, 'checkpoints/')
    checkpoints_path = tf.train.latest_checkpoint(checkpoints_dir)
    recovery_checpoints_dir = os.path.join(model_folder, 'recovery_checkpoints/')
    recovery_path = tf.train.latest_checkpoint(recovery_checpoints_dir)

    if checkpoints_path is not None:
        split = str.split(checkpoints_path, '/')
        name = split[-1]+'.meta'
        path = 'checkpoints/'
    elif recovery_path is not None:
        split = str.split(recovery_path, '/')
        name = split[-1]+'.meta'
        path = 'recovery_checkpoints/'
    else:
        print('No model found')
        return None
    
    return name, path


###############################################################################
###############################################################################
###############################################################################
###############################################################################

caffe_to_tf_layer_names = { 'encoder|conv1': 'encoder/conv1',
                            'encoder|conv2': 'encoder/conv2',
                            'encoder|conv3': 'encoder/conv3',
                            'encoder|conv3_1': 'encoder/conv3_1',
                            'encoder|conv4': 'encoder/conv4',
                            'encoder|conv4_1': 'encoder/conv4_1',
                            'encoder|conv5': 'encoder/conv5',
                            'encoder|conv5_1': 'encoder/conv5_1',
                            'encoder|conv6': 'encoder/conv6',
                            'encoder|conv6_1': 'encoder/conv6_1',
                            'features|conv1': 'features/conv1',
                            'features|conv2': 'features/conv2',
                            'features|conv3': 'features/conv3',
                            'encoder|conv_redir': 'encoder/conv_redir',
                            'conv_edges': 'decoder/conv_edges',
                            'encoder|predict|conv': 'encoder/predict/conv',
                            'decoder|refine_0|deconv': 'decoder/refine_0/deconv',
                            'decoder|refine_1|deconv': 'decoder/refine_1/deconv',
                            'decoder|refine_2|deconv': 'decoder/refine_2/deconv',
                            'decoder|refine_3|deconv': 'decoder/refine_3/deconv',
                            'decoder|refine_4|deconv': 'decoder/refine_4/deconv',
                            'decoder|refine_5|deconv': 'decoder/refine_5/deconv',
                            'decoder|refine_0|predict|conv': 'decoder/refine_0/predict/conv',
                            'decoder|refine_1|predict|conv': 'decoder/refine_1/predict/conv',
                            'decoder|refine_2|predict|conv': 'decoder/refine_2/predict/conv',
                            'decoder|refine_3|predict|conv': 'decoder/refine_3/predict/conv',
                            'decoder|refine_4|predict|conv': 'decoder/refine_4/predict/conv',
                            'decoder|refine_5|predict|conv': 'decoder/refine_5/predict/conv',
                            'decoder|refine_6|predict|conv': 'decoder/predict/conv',
                            'upsample_prediction1to0': 'decoder/refine_0/upsample_prediction/upsample_prediction1to0',
                            'upsample_prediction2to1': 'decoder/refine_1/upsample_prediction/upsample_prediction2to1',
                            'upsample_prediction3to2': 'decoder/refine_2/upsample_prediction/upsample_prediction3to2',
                            'upsample_prediction4to3': 'decoder/refine_3/upsample_prediction/upsample_prediction4to3',
                            'upsample_prediction5to4': 'decoder/refine_4/upsample_prediction/upsample_prediction5to4',
                            'upsample_prediction6to5': 'decoder/refine_5/upsample_prediction/upsample_prediction6to5'}


def load_caffe_weights(parameters_path):
    global parameter_names_list
    print(parameter_names_list)
    print('Loading caffe model:  '+parameters_path)
    data = h5py.File(parameters_path, 'r')['data']

    layers = {}

    for key in data.keys():
        if len(data[key].keys())==2:
            layers[key] = {parameter_names_list[0]: np.array(data[key]['0']), parameter_names_list[1]: np.array(data[key]['1'])}
    print()
    return layers
    



def load_caffe_activation(path, layer):
    r = tb.read(path+'/'+layer+'.float3')
    l = np.moveaxis(r, 2,0)
    return np.array([l]*1)      # 8 times only to fit the batch size, Eddy gave me one example




def convert_layer_weights(caffe_layer_weights):
    '''
    Rearrange dimensions of a single layer to be compatible with tensorflow kernels convention:
    
            CAFFE:  (#out, #in, h, w)
            TF:     (h, w, #in, #out)
    
    Return a numpy array in the tf convention containing the kernel weights
    '''
    tf_weights = np.moveaxis(caffe_layer_weights, 1, -1)  # move input dimension to last axis
    tf_weights = np.moveaxis(tf_weights, 0, -1)  # move output dimenstion to last axis
    
    return tf_weights




def convert_model_weights(weights_path='weights'):
    caffe_model_weights = load_caffe_weights(weights_path)
    ''' Convert the model to be compatible with tensorflow kernels, layer by layer '''
    converted = {}
    for w_key in caffe_model_weights:
        converted[w_key] = {parameter_names_list[0]: convert_layer_weights(caffe_model_weights[w_key][parameter_names_list[0]]),
                            parameter_names_list[1]: caffe_model_weights[w_key][parameter_names_list[1]]}
        # except:
        #     print('WARNING parameters conversion failed for: '+w_key)
    return converted




def weight_assignments(tf_trainable_variables, weights_path, caffe_to_tf_names_dict=None, parameter_names=['kernel', 'bias'], prefix=None):
    '''
    Prepare and return the variable assignment ops needed to load the caffe layers on the tensorflow model.
    The returned ops can be ran by the calling function.
    
    Arguments:
        tf_trainable_variables: list containing the tensorflow variables relative to the (trainable) layers of the model
        weights_path: path to the .npy file containing the pretrained weights in the dictionary format
    '''

    global parameter_names_list
    parameter_names_list = parameter_names

    caffe_dict = convert_model_weights(weights_path)       # convert caffe kernels into tf HWIO format
    ref_caffe_dict = caffe_dict.copy()
    assignment_ops = []
    
    tf_dict = dict(zip([var.name for var in tf_trainable_variables], tf_trainable_variables))   # create a dictionary of the tf layers
    for caffe_key in caffe_dict.keys():                                                         # try matching each caffe layer to a tf one
        try:
            if caffe_to_tf_names_dict is not None:
                tf_layer = caffe_to_tf_names_dict[caffe_key]
            else:
                if prefix is not None:
                    tf_layer = prefix+'/'+caffe_key.replace('|', '/')
                else:
                    tf_layer = caffe_key.replace('|', '/')
            try:
                for name in parameter_names_list:
                    print('* ' + caffe_key +' -> ' + tf_layer + ';  '+name)
                    assignment_ops.append(tf.assign(tf_dict[tf_layer + '/'+name+':0'], caffe_dict[caffe_key][name], name='assign_' + tf_layer + '_' +name))
                    del tf_dict[tf_layer + '/'+name+':0']
                # assignment_ops.append(tf.assign(tf_dict[tf_layer+'/kernel:0'], caffe_dict[caffe_key]['kernel'], name='assign_'+tf_layer+'_kernel'))
                # del tf_dict[tf_layer+'/kernel:0']
                # assignment_ops.append(tf.assign(tf_dict[tf_layer+'/bias:0'], caffe_dict[caffe_key]['bias'], name='assign_'+tf_layer+'_bias'))
                # del tf_dict[tf_layer+'/bias:0']
                del ref_caffe_dict[caffe_key]
            except KeyError:
                print('KeyError: layer \'' + tf_layer + '\', (translated from \''+caffe_key+'\' caffe layer), not found in known tensorflow layers')
        except KeyError:
            print('KeyError: layer \'' + caffe_key + '\' not found in know caffe layers')

    
    if len(ref_caffe_dict.keys())>0:
        print('Unmatched caffe layers:')
        [print('   \''+key+'\':') for key in ref_caffe_dict.keys()]
    # if len(tf_dict.keys())>0:
    #     print('Unmatched tensorflow layers:')
    #     [print('   '+str(tf_dict[key])) for key in tf_dict.keys()]
    
    
    return assignment_ops





def feed_dict_parameters(tf_trainable_variables, weights_path='weights'):
    '''
    Alternative to weight_assignments(...)
    The returned structure can be fed to session.run()
    
    Arguments:
        tf_trainable_variables: list containing the tensorflow variables relative to the (trainable) layers of the model
        weights_path: path to the .npy file containing the pretrained weights in the dictionary format
    '''
    
    caffe_dict = convert_model_weights(weights_path)       # convert caffe kernels into tf HWIO format
    ref_caffe_dict = caffe_dict.copy()
    feed_dict_params = {}
    
    tf_dict = dict(zip([var.name for var in tf_trainable_variables], tf_trainable_variables))   # create a dictionary of the tf layers
    for caffe_key in caffe_dict.keys():                                                         # try matching each caffe layer to a tf one
        try:
            tf_layer = caffe_to_tf_layer_names[caffe_key]
            try:
                feed_dict_params[tf_dict[tf_layer+'/kernel:0'].name] = caffe_dict[caffe_key]['kernel']
                del tf_dict[tf_layer+'/kernel:0']
                feed_dict_params[tf_dict[tf_layer+'/bias:0'].name] = caffe_dict[caffe_key]['bias']
                del tf_dict[tf_layer+'/bias:0']
                del ref_caffe_dict[caffe_key]
            except KeyError:
                print('KeyError: layer \'' + tf_layer + '\', (translated from \''+caffe_key+'\' caffe layer), not found in known tensorflow layers')
        except KeyError:
            print('KeyError: layer \'' + caffe_key + '\' not found in know caffe layers')

    
    if len(ref_caffe_dict.keys())>0:
        print('Unmatched caffe layers:')
        [print('   '+key) for key in ref_caffe_dict.keys()]
    if len(tf_dict.keys())>0:
        print('Unmatched tensorflow layers:')
        [print('   '+str(tf_dict[key])) for key in tf_dict.keys()]
    
    return feed_dict_params
