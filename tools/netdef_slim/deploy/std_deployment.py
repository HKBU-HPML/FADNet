import netdef_slim as nd
from math import ceil


class StandardDeployment:

    def __init__(self):
        self._width = None
        self._height = None
        self._temp_width = None
        self._temp_height = None
        self._rescale_coeff_x = None
        self._rescale_coeff_y = None
        self._scale = 1.0

    def input_image_resample(self, img):
        img_nomean = nd.ops.scale_and_subtract_mean(img)

        return nd.ops.resample(img_nomean,
            width=self._temp_width,
            height=self._temp_height,
            type='LINEAR',
            antialias=True
        )

    def input_resample_linear(self, data):
        return nd.ops.resample(data,
            width=self._temp_width,
            height=self._temp_height,
            type='LINEAR',
            antialias=True
        )

    def input_resample_nearest(self, data):
        return nd.ops.resample(data,
            width=self._temp_width,
            height=self._temp_height,
            type='NEAREST',
            antialias=True
        )

    def input_resample_flow(self, flow):
        return nd.ops.scale(
            nd.ops.resample(flow,
                width=self._temp_width,
                height=self._temp_height,
                reference=None,
                type='LINEAR',
                antialias=True
            ),
        (1.0/self._rescale_coeff_x, 1.0/self._rescale_coeff_y))

    def input_resample_disp(self, disp):
        return nd.ops.scale(
            nd.ops.resample(disp,
                width=self._temp_width,
                height=self._temp_height,
                reference=None,
                type='LINEAR',
                antialias=True
            ),1.0/self._rescale_coeff_x)


    def input_resample_binary(self, data):
        return nd.ops.threshold(
            nd.ops.resample(data,
                width=self._temp_width,
                height=self._temp_height,
                reference=None,
                type='LINEAR',
                antialias=True
            ),
        thresh=0.5)

    def output_resample_linear(self, data):
        return nd.ops.resample(data,
            width=self._width,
            height=self._height,
            type='LINEAR',
            antialias=True
        )

    def output_resample_nearest(self, data):
        return nd.ops.resample(data,
            width=self._width,
            height=self._height,
            type='NEAREST',
            antialias=True
        )

    def output_resample_flow(self, flow):
        return nd.ops.scale(
            nd.ops.resample(flow,
                width=self._width,
                height=self._height,
                reference=None,
                type='LINEAR',
                antialias=True
            ),
        (self._rescale_coeff_x, self._rescale_coeff_y))

    def output_resample_disp(self, disp):
        return nd.ops.scale(
            nd.ops.resample(disp,
                width=self._width,
                height=self._height,
                reference=None,
                type='LINEAR',
                antialias=True
            ), self._rescale_coeff_x)


    def output_resample_binary(self, data):
        return nd.ops.threshold(
            nd.ops.resample(data,
                width=self._width,
                height=self._height,
                reference=None,
                type='LINEAR',
                antialias=True
            ),
        thresh=0.5)

    def map_output(self, pred, output):
        self._rescale_coeff_x = self._width / self._temp_width
        self._rescale_coeff_y = self._height / self._temp_height
        pred.map('flow', self.output_resample_flow, output)
        pred.map('disp', self.output_resample_disp, output)
        pred.map('occ', self.output_resample_linear, output)
        pred.map('mb', self.output_resample_linear, output)
        pred.map('db', self.output_resample_linear, output)
        return output

    def make_graph(self, data, net_graph_constructor, divisor=64., scale=1.0):
        self._scale = scale
        self._width = data.width
        self._height = data.height

        self._temp_width = ceil(self._width*self._scale/divisor) * divisor
        self._temp_height = ceil(self._height*self._scale/divisor) * divisor
        self._rescale_coeff_x = self._width / self._temp_width
        self._rescale_coeff_y = self._height / self._temp_height

        nd.log('data:')
        nd.log(data)

        input = nd.Struct()

        data.map('img', self.input_image_resample, input)
        data.map('flow', self.input_resample_flow, input)
        data.map('disp', self.input_resample_disp, input)
        data.map('occ', self.input_resample_binary, input)
        data.map('mb', self.input_resample_binary, input)
        data.map('db', self.input_resample_binary, input)
        pred = net_graph_constructor(input)
        nd.log('pred:')
        nd.log(pred)

        if isinstance(pred, (tuple)):
            pred = pred[0]

        if isinstance(pred, (list)):
            output = []
            for prediction in pred:
                out = nd.Struct()
                self.map_output(prediction.final, out)
                output.append(out)

        else:
            output = nd.Struct()
            self.map_output(pred.final, output)

        nd.log('output:')
        nd.log(output)

        return output


