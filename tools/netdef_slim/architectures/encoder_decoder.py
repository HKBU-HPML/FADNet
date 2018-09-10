import netdef_slim as nd


class EncoderDecoderArchitecture:

    def __init__(self,
                 loss_function,
                 disassembling_function,
                 exit_after=2,
                 encoder_level=6,
                 interconv=False,
                 channel_factor=1.0,
                 num_outputs=2,
                 conv_upsample=False,
                 encoder_channels=None,
                 decoder_channels=None,
                 loss_weights=None):

        self._channel_factor = channel_factor
        self._encoder_channels = encoder_channels
        self._decoder_channels = decoder_channels
        self._loss_weights = loss_weights
        self._interconv = interconv
        self._loss_function = loss_function
        self._disassembling_function = disassembling_function
        self._num_outputs = num_outputs
        self._exit_after = exit_after
        self._conv_upsample = conv_upsample
        self._encoder_level = encoder_level

    def upsample_prediction(self, pred, ref, name):
        if self._conv_upsample:
            return nd.scope.upconv(pred, name=name, kernel_size=4, stride=2, pad=1, num_output=self._num_outputs)
        else:
            return nd.ops.resample(pred, reference=ref, type='LINEAR')

    def refine(self, level, input, features, out, input_prediction=None):
        num_output = self._decoder_channels['level%d' % level]

        with nd.Scope('refine_%d' % level):
            upconv = nd.scope.upconv_nl(input, name='deconv', kernel_size=4, stride=2, pad=1, num_output=num_output)

            concat_list = [features, upconv]
            if input_prediction is not None:
                upsampled_prediction = self.upsample_prediction(input_prediction, upconv, name="upsample_prediction%dto%d" % (level+1, level))
                concat_list.append(upsampled_prediction)

            concatenated = nd.ops.concat(concat_list, axis=1)

            if self._interconv:
                concatenated = nd.scope.conv_nl(
                    concatenated,
                    name="interconv",
                    kernel_size=3, stride=1, pad=1,
                    num_output=num_output)

            refined_prediction = self.predict(concatenated, level=level, loss_weight=self._loss_weights['level%d' % level], out=out)

            return concatenated, refined_prediction

    def predict(self, input, level, loss_weight, out):
        with nd.Scope('predict'):
            predicted = nd.scope.conv(input, name='conv', kernel_size=3, stride=1, pad=1, num_output=self._num_outputs)

            out.levels.make_struct(level)
            if callable(self._disassembling_function):
                out.levels[level] = self._disassembling_function(predicted)

            if callable(self._loss_function):
                self._loss_function(out.levels[level], level=level, weight=loss_weight)

            return predicted

    def make_graph(self, input, edge_features=None):
        raise NotImplementedError
