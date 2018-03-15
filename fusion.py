from dataset import load_pfm, save_pfm
import numpy as np
import sys

SGM_ROOT="data_local/dispnet/real_release/sgm_disp/left/"
NN_ROOT="real_detect_result_s1.5/"
imgNo=sys.argv[1]
SGM_NAME="%s.pfm" % imgNo
NN_NAME="predict_real_release_frames_cleanpass_%s.pfm" % imgNo

# f = open("real_sgm_release.list", "r")
# real_list = f.readlines()
# f.close()

sgm_np, scale = load_pfm(SGM_ROOT + SGM_NAME)
nn_np, scale = load_pfm(NN_ROOT + NN_NAME)

# transfer sgm_np to 0/1 map
sgm_np = np.minimum(sgm_np, 1)

# filter nn_np with sgm_np
nn_np = np.multiply(sgm_np, nn_np)

save_pfm("test.pfm", nn_np)


