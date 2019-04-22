from __future__ import print_function

from networks.simple_net import SimpleNet
from networks.dispnet_corr2 import DispNetCorr2
from networks.dispnet_v2 import DispNetCSRes, DispNetC

from settings import logger

SUPPORT_NETS = {'simplenet': SimpleNet,
        'dispnetcres': DispNetCSRes,
        'dispnetc': DispNetC,
        'dispnetcorr2': DispNetCorr2,}

def build_net(net_name):
    net  = SUPPORT_NETS.get(net_name, None)
    if net is None:
        logger.error('Current supporting nets: %s , Unsupport net: %s', SUPPORT_NETS.keys(), net_name)
        raise 'Unsupport net: %s' % net_name
    return net
