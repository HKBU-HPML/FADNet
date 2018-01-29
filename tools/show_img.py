import os
from subprocess import Popen

ORIGINAL_DATAPATH = '/media/sf_Shared_Data/gpuhomedataset/FlyingThings3D_release/disparity/TEST'
PREDICT_DATAPATH = '/media/sf_Shared_Data/gpuhome/repositories/pytorch-dispnet/detect_result_cd'
BIN = 'jview'
result_name = 'predict_A_0009_0011.pfm'


def _get_view_cmd(filepath):
    cmd = '{} {}'.format(BIN, filepath)
    return cmd

def _execute(cmd):
    p = Popen(cmd.split(' '))
    return p
    #os.system(cmd)

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

