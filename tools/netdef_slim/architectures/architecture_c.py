import netdef_slim as nd
from copy import copy
from .encoder_decoder import EncoderDecoderArchitecture


default_feature_channels = {
    'conv1': 64,
    'conv2': 128,
    'conv3': 256,
}


class Features_C:

    def __init__(self,
                 exit_after=3,
                 channel_factor=1.0,
                 feature_channels=None):

        self._channel_factor = channel_factor

        if feature_channels is None:
            self._feature_channels = {}
            for name, channels in default_feature_channels.items():
                self._feature_channels[name] = int(self._channel_factor*channels)
        else:
            self._feature_channels = feature_channels

        self._exit_after = exit_after

    def make_graph(self, img0, img1):
        data = nd.Struct()

        (data.conv1a, data.conv1b) = nd.scope.conv_nl((img0, img1),                name='conv1', kernel_size=7, stride=2, pad=3, num_output=self._feature_channels['conv1'])
        if self._exit_after == 1: return data

        (data.conv2a, data.conv2b) = nd.scope.conv_nl((data.conv1a, data.conv1b),  name='conv2', kernel_size=5, stride=2, pad=2, num_output=self._feature_channels['conv2'])
        if self._exit_after == 2: return data

        (data.conv3a, data.conv3b) = nd.scope.conv_nl((data.conv2a, data.conv2b),  name='conv3', kernel_size=5, stride=2, pad=2, num_output=self._feature_channels['conv3'])
        if self._exit_after == 3: return data

        return data


class Features_C_Mapper:

    def __init__(self,
                 exit_after=3,
                 feature_channels=None,
                 kernel_sizes=None,
                 strides=None):

        self._feature_channels = feature_channels
        self._kernel_sizes = kernel_sizes
        self._strides = strides

        if self._feature_channels is None:
            self._feature_channels = [64, 128, 256]
        if self._kernel_sizes is None:
            self._kernel_sizes = [7, 5, 5]
        if self._strides is None:
            self._strides = [2, 2, 2]

        self._exit_after = exit_after

    def make_graph(self, feat0, feat1):
        data = nd.Struct()

        (data.conv1a, data.conv1b) = nd.scope.conv_nl((feat0, feat1),              name='conv1_map', kernel_size=self._kernel_sizes[0], stride=self._strides[0], pad=int(self._kernel_sizes[0]/2), num_output=int(self._feature_channels[0]))
        if self._exit_after == 1: return data

        (data.conv2a, data.conv2b) = nd.scope.conv_nl((data.conv1a, data.conv1b),  name='conv2_map', kernel_size=self._kernel_sizes[1], stride=self._strides[1], pad=int(self._kernel_sizes[1]/2), num_output=int(self._feature_channels[1]))
        if self._exit_after == 2: return data

        (data.conv3a, data.conv3b) = nd.scope.conv_nl((data.conv2a, data.conv2b),  name='conv3_map', kernel_size=self._kernel_sizes[2], stride=self._strides[2], pad=int(self._kernel_sizes[2]/2), num_output=int(self._feature_channels[2]))
        if self._exit_after == 3: return data

        return data


default_encoder_channels = {
    'conv_redir': 32,
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


class Architecture_C_upper(EncoderDecoderArchitecture):

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

    def make_graph(self, input, edge_features=None, use_1D_corr=False, single_direction=0):
        out = nd.Struct()
        out.make_struct('levels')
        if use_1D_corr:
            corr = nd.ops.correlation_1d(input.conv3a, input.conv3b,
                        kernel_size=1,
                        max_displacement=40,
                        pad=40,
                        stride_1=1,
                        stride_2=1,
                        single_direction=single_direction
                        )
        else:
            corr = nd.ops.correlation_2d(input.conv3a, input.conv3b,
                        kernel_size=1,
                        max_displacement=20,
                        pad=20,
                        stride_1=1,
                        stride_2=2)


        with nd.Scope('encoder'):

            redir = nd.scope.conv_nl(input.conv3a,  name="conv_redir", kernel_size=1, stride=1, pad=0, num_output=self._encoder_channels['conv_redir'])
            merged = nd.ops.concat(redir, corr)

            conv3_1          = nd.scope.conv_nl(merged,  name="conv3_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv3_1'])
            conv4            = nd.scope.conv_nl(conv3_1, name="conv4",   kernel_size=3, stride=2, pad=1, num_output=self._encoder_channels['conv4'])
            conv4_1          = nd.scope.conv_nl(conv4,   name="conv4_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv4_1'])
            conv5            = nd.scope.conv_nl(conv4_1, name="conv5",   kernel_size=3, stride=2, pad=1, num_output=self._encoder_channels['conv5'])
            conv5_1          = nd.scope.conv_nl(conv5,   name="conv5_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv5_1'])
            conv6            = nd.scope.conv_nl(conv5_1, name="conv6",   kernel_size=3, stride=2, pad=1, num_output=self._encoder_channels['conv6'])
            conv6_1          = nd.scope.conv_nl(conv6,   name="conv6_1", kernel_size=3, stride=1, pad=1, num_output=self._encoder_channels['conv6_1'])

            prediction6        = self.predict(conv6_1, level=6, loss_weight=self._loss_weights['level6'], out=out)

        with nd.Scope('decoder'):

            decoder5, prediction5 = \
                self.refine(level=5,
                            input=conv6_1,
                            input_prediction=prediction6,
                            features=conv5_1, out=out)

            if self._exit_after == 5:
                out.final = out.levels[5]
                return out

            decoder4, prediction4 = \
                self.refine(level=4,
                            input=decoder5,
                            input_prediction=prediction5,
                            features=conv4_1, out=out)

            if self._exit_after == 4:
                out.final = out.levels[4]
                return out

            decoder3, prediction3 = \
                self.refine(level=3,
                            input=decoder4,
                            input_prediction=prediction4,
                            features=conv3_1, out=out)

            if self._exit_after == 3:
                out.final = out.levels[3]
                return out

            decoder2, prediction2 = \
                self.refine(level=2,
                            input=decoder3,
                            input_prediction=prediction3,
                            features=input.conv2a, out=out)

            if self._exit_after == 2:
                out.final = out.levels[2]
                return out

            decoder1, prediction1 = \
                self.refine(level=1,
                            input=decoder2,
                            input_prediction=prediction2,
                            features=input.conv1a, out=out)

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


class Architecture_C:

    def __init__(self, **kwargs):

        features_args = {}
        if 'channel_factor' in kwargs: features_args['channel_factor'] = kwargs['channel_factor']
        if 'feature_channels' in kwargs: features_args['feature_channels'] = kwargs['feature_channels']

        self._features = Features_C(**features_args)
        self._learn_features = kwargs.pop('learn_features', False)
        kwargs.pop('feature_channels', None)

        self._upper = Architecture_C_upper(**kwargs)

    def make_graph(self, img0, img1, edge_features=None, use_1D_corr=False, single_direction=0):
        with nd.Scope('features', learn=self._learn_features):
            feat = self._features.make_graph(img0, img1)

        with nd.Scope('upper'):
            out = self._upper.make_graph(feat, edge_features, use_1D_corr, single_direction)
            out.feat = feat
            return out

