//
//  lmbspecialops - a collection of tensorflow ops
//  Copyright (C) 2017  Albert Ludwigs University of Freiburg, Pattern Recognition and Image Processing, Computer Vision Group
//  Author(s): Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include "config.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

//#include <lz4.h>
#include <webp/decode.h>

using namespace tensorflow;

REGISTER_OP("DecodeWebp")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Output("image: uint8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

        DimensionHandle channels_dim;
        int32 channels;
        TF_RETURN_IF_ERROR(c->GetAttr("channels", &channels));
        if (channels == 0) {
          channels_dim = c->UnknownDim();
        } else {
          if (channels < 0) {
            return errors::InvalidArgument("channels must be non-negative, got ",
                                           channels);
          }
          channels_dim = c->MakeDim(channels);
        }

        c->set_output(0,
                      c->MakeShape({InferenceContext::kUnknownDim,
                                    InferenceContext::kUnknownDim, channels_dim}));

        return Status::OK();
    })
    .Doc(R"doc(
Decode a WEBP-encoded image to a uint8 or uint16 tensor.
contents: 0-D.  The WEBP-encoded image.
image: uint8 tensor with shape `[height, width, {3|4}]`.
channels: Number of expected channels (0:auto|3|4).
)doc");


// Decode the contents of a WEBP file
class DecodeWebpOp : public OpKernel {
 public:
  explicit DecodeWebpOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("channels", &channels_));
        OP_REQUIRES(
                  context,
                  channels_ == 0 || channels_ == 3 || channels_ == 4,
                  errors::InvalidArgument("channels must be 0, 3, or 4, got ",
                                          channels_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));

    // Start decoding image to get shape details
    const StringPiece data = contents.scalar<string>()();

    WebPBitstreamFeatures features;

    OP_REQUIRES(context, WebPGetFeatures((uint8_t*)data.data(), data.size(), &features) == VP8_STATUS_OK,
        errors::InvalidArgument("Invalid WEBP file"));

    int channels = channels_;
    if (channels == 0) {
      channels = features.has_alpha?4:3;
    }

    // Allocate tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({features.height, features.width, channels}), &output));

    // Decode webp
    WebPDecoderConfig config;
    OP_REQUIRES(context, WebPInitDecoderConfig(&config),
            errors::InvalidArgument("Invalid WEBP file"));

    config.options.bypass_filtering = 1;
    config.options.no_fancy_upsampling = 1;

    std::string tmp_data;
    tmp_data.resize(features.width * features.height * channels);

    config.output.colorspace = (channels == 4?MODE_RGBA:MODE_RGB);
    config.output.u.RGBA.rgba = output->flat<uint8>().data();
    config.output.u.RGBA.stride = features.width * channels;
    config.output.u.RGBA.size = config.output.u.RGBA.stride * features.height;
    config.output.is_external_memory = 1;
    OP_REQUIRES(context, WebPDecode((const uint8_t*)data.data(), data.size(), &config) == VP8_STATUS_OK,
                errors::InvalidArgument("Invalid WEBP file"));

    WebPFreeDecBuffer(&config.output);
  }

 private:
  int channels_;
};
REGISTER_KERNEL_BUILDER(Name("DecodeWebp").Device(DEVICE_CPU), DecodeWebpOp);

