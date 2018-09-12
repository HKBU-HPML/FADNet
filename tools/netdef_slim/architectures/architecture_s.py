import netdef_slim as nd
from .encoder_decoder import EncoderDecoderArchitecture
from copy import copy

default_encoder_channels = {
    'conv1': 64,
    'conv2': 128,
    'conv3': 256,
    'conv3_1': 256,
    'conv4': 512,
    'conv4_1': 512,
    'conv5': 512,
    'conv5_1': 512,
    'conv6': 1024,
    'conv6_1': 1024
}


default_decoder_channels = {
    'level5': 512,
    'level4': 256,
    'level3': 128,
    'level2': 64,
    'level1': 32,
    'level0': 16
}


default_loss_weights = {
    'level6': 1/16,
    'level5': 1/16,
    'level4': 1/16,
    'level3': 1/8,
    'level2': 1/4,
    'level1': 1/2,
    'level0': 1/1
}


class Architecture_S(EncoderDecoderArchitecture):
    
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

    def make_graph(self, input, edge_features=None):
        out = nd.Struct()
        out.make_struct('levels')

        with nd.Scope('encoder'):
            conv1            = nd.scope.conv_nl(input,   name="conv1",   kernel_size=7, stride=2, pad=3, num_output=self._encoder_channels['conv1'])
            conv2            = nd.scope.conv_nl(conv1,   name="conv2",   kernel_size=5, stride=2, pad=2, num_output=self._encoder_channels['conv2'])
            conv3            = nd.scope.conv_nl(conv2,   name="conv3",   kernel_size=5, stride=2, pad=2, num_output=self._encoder_channels['conv3'])
            conv3_1          = nd.scope.conv_nl(conv3,   name="conv3_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv3_1'])
            conv4            = nd.scope.conv_nl(conv3_1, name="conv4",   kernel_size=3, stride=2, pad=1, num_output=self._encoder_channels['conv4'])
            conv4_1          = nd.scope.conv_nl(conv4,   name="conv4_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv4_1'])
            if self._encoder_level == 4:
                prediction4 = self.predict(conv4_1, level=4, loss_weight=self._loss_weights['level4'], out=out)
            else:
                conv5            = nd.scope.conv_nl(conv4_1, name="conv5",   kernel_size=3, stride=2, pad=1, num_output=self._encoder_channels['conv5'])
                conv5_1          = nd.scope.conv_nl(conv5,   name="conv5_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv5_1'])
                if self._encoder_level == 5:
                    prediction5 = self.predict(conv5_1, level=5, loss_weight=self._loss_weights['level5'], out=out)
                else:
                    conv6            = nd.scope.conv_nl(conv5_1, name="conv6",   kernel_size=3, stride=2, pad=1, num_output=self._encoder_channels['conv6'])
                    conv6_1          = nd.scope.conv_nl(conv6,   name="conv6_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv6_1'])
            
                    prediction6        = self.predict(conv6_1, level=6, loss_weight=self._loss_weights['level6'], out=out)

        with nd.Scope('decoder'):

            if self._encoder_level >= 6:
                decoder5, prediction5 = \
                    self.refine(level=5,
                                input=conv6_1,
                                input_prediction=prediction6,
                                features=conv5_1, out=out)

                if self._exit_after == 5:
                    out.final = out.levels[5]
                    return out

            if self._encoder_level >= 5:
                decoder4, prediction4 = \
                    self.refine(level=4,
                                input=decoder5 if self._encoder_level > 5 else conv5_1,
                                input_prediction=prediction5,
                                features=conv4_1, out=out)

                if self._exit_after == 4:
                    out.final = out.levels[4]
                    return out

            decoder3, prediction3 = \
                self.refine(level=3,
                            input=decoder4 if self._encoder_level > 4 else conv4_1,
                            input_prediction=prediction4,
                            features=conv3_1, out=out)

            if self._exit_after == 3:
                out.final = out.levels[3]
                return out

            decoder2, prediction2 = \
                self.refine(level=2,
                            input=decoder3,
                            input_prediction=prediction3,
                            features=conv2, out=out)

            if self._exit_after == 2:
                out.final = out.levels[2]
                return out

            decoder1, prediction1 = \
                self.refine(level=1,
                            input=decoder2,
                            input_prediction=prediction2,
                            features=conv1, out=out)

            if self._exit_after == 1:
                out.final = out.levels[1]
                return out

            if edge_features is None:
                raise BaseException('Architecture_S needs edge features if now shallow')

            edges = nd.scope.conv_nl(edge_features,
                                name="conv_edges",
                                kernel_size=3,
                                stride=1,
                                pad=1,
                                num_output=self._decoder_channels['level0'])

            decoder0, prediction0 = \
                self.refine(level=0,
                            input=decoder1,
                            input_prediction=prediction1,
                            features=edges, out=out)

            out.final = out.levels[0]
            return out


    


