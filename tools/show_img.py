import os
from subprocess import Popen

#ORIGINAL_DATAPATH = '/media/sf_Shared_Data/gpuhomedataset/FlyingThings3D_release/disparity/TEST'
ORIGINAL_DATAPATH = '/media/sf_Shared_Data/gpuhomedataset/clean_dispnet/FlyingThings3D_release/disparity/TEST'
#PREDICT_DATAPATH = '/media/sf_Shared_Data/gpuhome/repositories/pytorch-dispnet/detect_result_cd'
PREDICT_DATAPATH = '/media/sf_Shared_Data/gpuhome/repositories/pytorch-dispnet/cc_detect_result'
BIN = 'jview'
#result_name = 'predict_A_0019_0015.pfm'
#result_name = 'predict_A_0011_0007.pfm'
#result_name = 'predict_A_0009_0014.pfm'
#result_name = 'predict_A_0011_0012.pfm'
#result_name = 'predict_A_0011_0015.pfm'
result_name = 'predict_A_0001_0015.pfm'


def _get_view_cmd(filepath):
    cmd = '{} {}'.format(BIN, filepath)
    return cmd

def _execute(cmd):
    p = Popen(cmd.split(' '))
    return p
    #os.system(cmd)

def show_img(filepath):
    return _execute(_get_view_cmd(filepath))

def show_images(result_name):
    show_original_cmd = _get_view_cmd(os.path.join(PREDICT_DATAPATH, result_name))
    ps = []
    p = _execute(show_original_cmd)
    ps.append(p)
    name_items = result_name.split('_')
    left_image_path = os.path.join(ORIGINAL_DATAPATH, name_items[1], name_items[2], 'left', name_items[3])
    #right_image_path = os.path.join(ORIGINAL_DATAPATH, name_items[1], name_items[2], 'right', name_items[3])
    left_cmd = _get_view_cmd(left_image_path)
    #right_cmd = _get_view_cmd(right_image_path)
    p = _execute(left_cmd)
    ps.append(p)
    #p = _execute(right_cmd)
    #ps.append(p)
    raw_input("Press Enter to continue...")
    for p in ps:
        p.terminate()


if __name__ == '__main__':
    show_images(result_name)

