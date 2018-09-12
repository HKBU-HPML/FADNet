from __future__ import print_function
import argparse
import os, shutil, sys, gc
import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from dataset import *
#from dispnet_v2 import *
from dispnet import *
from multiscaleloss import *
#from monodepthloss import *
from lr_monoloss import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import csv
from monodepth_net import resnet50_decoder

dispnet = DispNetCSResWithMono(4, False, True, input_channel=3)
own_state = dispnet.state_dict()

origin_disp = "models/flying-real-dispCSR-in1024/dispS_epoch_36.pth"
model_data = torch.load(origin_disp)['state_dict']
#print(model_data.keys())
for key in model_data.keys():
    own_state[key] = model_data[key]

origin_mono = "models/flying-real-dispCSR-in1024/mono_epoch_36.pth"
model_data = torch.load(origin_mono)['state_dict']
#print(model_data.keys())

for key in model_data.keys():
    print(key)
    own_key = 'dispnetc.mono_net.%s' % key
    own_state[own_key] = model_data[key]
    #model_data[key] = own_state[key]

dispnet.load_state_dict(own_state)
torch.save({'epoch': 0,
            'arch': 'dispnetcsres_mono',
            'state_dict':own_state,
            'best_EPE': 1.0,}
            , 'dispnetcsres-mono.pth')
