import json, yaml
import logging

def load_loss_scheme(loss_config):

    with open(loss_config, 'r') as f:
        loss_json = yaml.safe_load(f)

    return loss_json

DEBUG =0
logger = logging.getLogger()

if DEBUG:
    #coloredlogs.install(level='DEBUG')
    logger.setLevel(logging.DEBUG)
else:
    #coloredlogs.install(level='INFO')
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)


#from netdef_slim.utils.io import read 
#left_img = sys.argv[1]
#subfolder = sys.argv[2]
#
#occ_file = 'tmp/disp.L.float3'
#occ_data = read(occ_file) # returns a numpy array
#
#import matplotlib.pyplot as plt
#occ_data = occ_data[::-1, :, :] * -1.0
#print(np.mean(occ_data))
##plt.imshow(occ_data[:,:,0], cmap='gray')
## plt.show()
#
#subfolder = "detect_results/%s" % subfolder
#if not os.path.exists(subfolder):
#    os.makedirs(subfolder)
#
##name_items = left_img.split('.')[0].split('/')
##save_name = '_'.join(name_items) + '.pfm'
#name_items = left_img.split('/')
#filename = name_items[-1]
#topfolder = name_items[-2]
#save_name = filename + '.pfm'
#target_folder = '%s/%s' % (subfolder, topfolder)
#print('target_folder: ', target_folder)
#if not os.path.exists(target_folder):
#    os.makedirs(target_folder)
#save_pfm('%s/%s' % (target_folder, save_name), occ_data[:,:,0])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
