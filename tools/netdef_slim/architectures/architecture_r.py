import netdef_slim as nd
from .encoder_decoder import EncoderDecoderArchitecture
from copy import copy

default_encoder_channels = {
    'conv0': 64,
    'conv1': 64,
    'conv1_1': 128,
    'conv2': 128,
    'conv2_1': 128,
}

default_decoder_channels = {
    'level1': 32,
    'level0': 16
}


default_loss_weights = {
    'level2': 0,
    'level1': 0,
    'level0': 1.0
}


class Architecture_R(EncoderDecoderArchitecture):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._encoder_channels is None:
            self._encoder_channels = {}
            for name, channels in default_encoder_channels.items():
                self._encoder_channels[name] = int(self._channel_factor*channels)

        if self._decoder_channels is None:
            self._decoder_channels = {}
            for name, channels in default_decoder_channels.items():
                self._decoder_channels[name] = int(self._channel_factor*channels)

        if self._loss_weights is None:
            self._loss_weights = copy(default_loss_weights)

    def make_graph(self, input):
        out = nd.Struct()
        out.make_struct('levels')

        with nd.Scope('encoder'):
            conv0            = nd.scope.conv_nl(input,   name="conv0",   kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv0'])
            conv1            = nd.scope.conv_nl(conv0,   name="conv1",   kernel_size=3, stride=2, pad=1, num_output=self._encoder_channels['conv1'])
            conv1_1          = nd.scope.conv_nl(conv1,   name="conv1_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv1_1'])
            conv2            = nd.scope.conv_nl(conv1_1, name="conv2",   kernel_size=3, stride=2, pad=1, num_output=self._encoder_channels['conv2'])
            conv2_1          = nd.scope.conv_nl(conv2,   name="conv2_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv2_1'])

            prediction2 = self.predict(conv2_1, level=2, loss_weight=self._loss_weights['level2'], out=out)

        with nd.Scope('decoder'):
            decoder1, prediction1 = \
                self.refine(level=1,
                            input=conv2_1,
                            input_prediction=prediction2,
                            features=conv1_1, out=out)

            if self._exit_after == 1:
                out.final = out.levels[1]
                return out

            decoder0, prediction0 = \
                self.refine(level=0,
                            input=decoder1,
                            input_prediction=prediction1,
                            features=conv0, out=out)

            out.final = out.levels[0]
            return out



