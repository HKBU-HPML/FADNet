import netdef_slim as nd

class BaseNetwork:
    def __init__(self, pretrain=False, scale=1.0, conv_upsample=False, batch_norm=False, channel_factor=1.0, feature_channels=None):
        super().__init__()

        self._scale = scale
        self._conv_upsample = conv_upsample
        self._batch_norm = batch_norm
        self._channel_factor = channel_factor
        self._feature_channels = feature_channels

    def make_graph(self, data):
        raise NotImplementedError

    def scope_args(self):
        return {
          'conv_nonlin_op': nd.ops.conv_relu if not self._batch_norm else nd.ops.conv_bn_relu,
          'upconv_nonlin_op': nd.ops.upconv_relu if not self._batch_norm else nd.ops.upconv_bn_relu
        }
