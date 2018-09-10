import netdef_slim as nd

class BaseEnvironment:
    def __init__(self, net):
        self._net = net
        self._deploy_divisor = 64.0

    def make_net_graph(self, data, include_losses=True, apply_mappers=False):
        raise NotImplementedError

    def make_deploy_graph(self, data, scale):

        output = self.translate_output(StandardDeployment().make_graph(
            data=data,
            net_graph_constructor=lambda data: self.make_net_graph(data, include_losses=False),
            divisor=self._deploy_divisor,
            scale=scale
        ))
        return output
