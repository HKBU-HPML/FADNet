from __future__ import print_function

from networks.DispNetC import DispNetC
from networks.DispNetS import DispNetS
from networks.DispNetCS import DispNetCS
from networks.DispNetCSS import DispNetCSS
from networks.FADNet import FADNet
#from networks.MobileFADNet3 import MobileFADNet
from networks.stackhourglass import PSMNet
from networks.GANet_deep import GANet
from networks.gwcnet import GwcNet
from networks.aanet import AANet
from utils.common import logger

SUPPORT_NETS = {
        'fadnet': FADNet,
        'mobilefadnet': FADNet,
        'slightfadnet': FADNet,
        'tinyfadnet': FADNet,
        'microfadnet': FADNet,
        'xfadnet': FADNet,
        'crl': FADNet,
        'dispnetc': DispNetC,
        'dispnets': DispNetS,
        'dispnetcs': DispNetCS,
        'dispnetcss': DispNetCSS,
        'psmnet': PSMNet,
        'ganet':GANet,
        'gwcnet':GwcNet,
        'aanet':AANet
        }

def build_net(net_name):
    net  = SUPPORT_NETS.get(net_name, None)
    if net is None:
        logger.error('Current supporting nets: %s , Unsupport net: %s', SUPPORT_NETS.keys(), net_name)
        raise 'Unsupport net: %s' % net_name
    return net
