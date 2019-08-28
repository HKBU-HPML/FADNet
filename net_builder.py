from __future__ import print_function

#from networks.simple_net import SimpleNet
#from networks.dispnet_corr2 import DispNetCorr2
from networks.DispNetC import DispNetC
from networks.DispNetS import DispNetS
from networks.DispNetCSS import DispNetCSS
from networks.DispNetCSRes import DispNetCSRes
from networks.stackhourglass import PSMNet
from networks.GANet_deep import GANet
#from networks.MultiCorrNet import MultiCorrNet

from utils.common import logger

SUPPORT_NETS = {
        #'simplenet': SimpleNet,
        'dispnetcres': DispNetCSRes,
        'dispnetc': DispNetC,
        'dispnets': DispNetS,
        'dispnetcss': DispNetCSS,
        'psmnet': PSMNet,
        'ganet':GANet,
        #'multicorrnet': MultiCorrNet,
        #'dispnetcorr2': DispNetCorr2,
        }

def build_net(net_name):
    net  = SUPPORT_NETS.get(net_name, None)
    if net is None:
        logger.error('Current supporting nets: %s , Unsupport net: %s', SUPPORT_NETS.keys(), net_name)
        raise 'Unsupport net: %s' % net_name
    return net
