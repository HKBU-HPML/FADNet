import netdef_slim as nd
from netdef_slim.networks.base_env import BaseEnvironment
from netdef_slim.translators import *
from netdef_slim.deploy import StandardDeployment


class FlowNet2f_Environment(BaseEnvironment):
    def __init__(self, net, scale=1.0, conv_upsample=False):
        super().__init__(net)
        self._scale = scale
        self._conv_upsample = conv_upsample


    def make_net_graph(self, data, **kwargs):
        return self._net.make_graph(data, **kwargs)

    def make_eval_graph(self, width, height, scale=1.0):
        data = nd.Struct()
        data.width = width
        data.height = height
        data.img = nd.Struct()
        data.img[0] = nd.placeholder('data.img0', (1, 3, height, width))
        data.img[1] = nd.placeholder('data.img1', (1, 3, height, width))

        output = self.translate_output(StandardDeployment().make_graph(
            data=data,
            net_graph_constructor=lambda data: self.make_net_graph(data, include_losses=False),
            divisor=self._deploy_divisor
        ))

        return output


    def translate_output(self, data):
        data2 = nd.Struct()
        data.copy(data2)

        data2.translate('mb', 'mb_soft', softmax2_soft_translator)
        data2.translate('mb', 'mb', softmax2_hard_translator)
        data2.translate('occ', 'occ_soft', softmax2_soft_translator)
        data2.translate('occ', 'occ', softmax2_hard_translator)

        return data2
