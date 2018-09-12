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
#include "tensorflow/core/platform/cpu_info.h"

using namespace tensorflow;

REGISTER_OP("DecodePfm")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Output("image: float32")
    .Output("scale: float32")
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
      c->set_output(1, c->Scalar());

      return Status::OK();
    })
    .Doc(R"doc(
Decode a PFM-encoded image to a uint8 or uint16 tensor.
contents: 0-D.  The PFM-encoded image.
channels: Number of color channels for the decoded image.
image: 3-D with shape `[height, width, {3|1}]`.
scale: Scalar containing the scale.
)doc");




// Decode the contents of a PFM file
class DecodePfmOp : public OpKernel {
 public:
  explicit DecodePfmOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("channels", &channels_));
    OP_REQUIRES(
              context,
              channels_ == 0 || channels_ == 1 || channels_ == 3,
              errors::InvalidArgument("channels must be 0, 1, or 3, got ",
                                      channels_));
  }

  static bool skipWhitespace(OpKernelContext* context, const StringPiece& data, size_t* index, bool onlyOne = false) {
    if ((*index) >= data.size()) {
      context->CtxFailure(errors::InvalidArgument("Invalid PFM header, unexpected end of file"));
      return false;
    }
    char c = data[(*index)];
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
      context->CtxFailure(errors::InvalidArgument("Invalid PFM header, expected float at ", (*index)));
      return false;
    }
    (*index)++;
    if (onlyOne) {
      return true;
    }
    for (; (*index) < data.size(); (*index)++) {
        c = data[(*index)];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            break;
        }
    }
    return true;
  }

  static bool readInt(OpKernelContext* context, const StringPiece& data, size_t* index, uint32* number) {
    if ((*index) >= data.size()) {
      context->CtxFailure(errors::InvalidArgument("Invalid PFM header, unexpected end of file"));
      return false;
    }
    char c = data[(*index)];
    if (c < '0' || c > '9') {
      context->CtxFailure(errors::InvalidArgument("Invalid PFM header, expected float at ", (*index)));
      return false;
    }
    *number = 0;
    for (; (*index) < data.size(); (*index)++) {
      c = data[(*index)];
      if (c >= '0' && c <= '9') {
        *number *= 10;
        *number += c - '0';
      } else {
          break;
      }
    }
    return true;
  }

  static bool readFloat(OpKernelContext* context, const StringPiece& data, size_t* index, float* number) {
    if ((*index) >= data.size()) {
      context->CtxFailure(errors::InvalidArgument("Invalid PFM header, unexpected end of file"));
      return false;
    }
    char c = data[(*index)];
    bool neg = false;
    if (c == '-' || c == '+') {
        neg = (c == '-');
        (*index)++;
        if ((*index) >= data.size()) {
          context->CtxFailure(errors::InvalidArgument("Invalid PFM header, unexpected end of file"));
          return false;
        }
        c = data[(*index)];
    }
    if (c < '0' || c > '9') {
      context->CtxFailure(errors::InvalidArgument("Invalid PFM header, expected float at ", (*index)));
      return false;
    }
    *number = 0;
    for (; (*index) < data.size(); (*index)++) {
      c = data[(*index)];
      if (c == '.') {
        break;
      }
      if (c >= '0' && c <= '9') {
        *number *= 10;
        *number += c - '0';
      } else {
          break;
      }
    }
    if (c == '.') {
      (*index)++;
      float factor = 0.1f;
      for (; (*index) < data.size(); (*index)++) {
        c = data[(*index)];
        if (c >= '0' && c <= '9') {
          *number += (c - '0') * factor;
          factor *= 0.1f;
        } else {
            break;
        }
      }
    }
    if (neg) {
      *number = -*number;
    }
    return true;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));

    // Start decoding image to get shape details
    const StringPiece data = contents.scalar<string>()();
    if (data.size() < 9) {
      //Pf<whitespace>0<whitespace>0<whitespace>0<whitespace><data>
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Invalid PFM data size, data too small for PFM file"));
    }
    int channels = channels_;
    if (data.starts_with("PF")) {
        OP_REQUIRES(context, channels == 0 || channels == 3,
                      errors::InvalidArgument("File has 3 channels, but output has only 1"));
        channels = 3;
    } else if (data.starts_with("Pf")) {
        OP_REQUIRES(context, channels == 0 || channels == 1,
                          errors::InvalidArgument("File has 1 channels, but output has 3"));
        channels = 1;
    } else {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Invalid PFM header, expected 'P6'"));
    }
    size_t index = 2;
    if (!skipWhitespace(context, data, &index)) {
      return;
    }
    uint32 width = 0, height = 0;
    float scale = 0.0f;
    if (!readInt(context, data, &index, &width)) {
      return;
    }
    if (!skipWhitespace(context, data, &index)) {
      return;
    }
    if (!readInt(context, data, &index, &height)) {
      return;
    }
    if (!skipWhitespace(context, data, &index)) {
      return;
    }
    if (!readFloat(context, data, &index, &scale)) {
      return;
    }
    if (!skipWhitespace(context, data, &index, true)) {
      return;
    }
    // Verify that width and height are not too large:
    // - verify width and height don't overflow int.
    // - width can later be multiplied by channels_ and sizeof(uint16), so
    //   verify single dimension is not too large.
    // - verify when width and height are multiplied together, there are a few
    //   bits to spare as well.
    const int64 total_size =
                static_cast<int64>(width) * static_cast<int64>(height);
    if (width != static_cast<int64>(width) || width <= 0 ||
        width >= (1LL << 27) || height != static_cast<int64>(height) ||
        height <= 0 || height >= (1LL << 27) || total_size >= (1LL << 29)) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("PFM size too large for int: ",
                                          width, " by ", height));
    }

    bool littleEndian = false;
    if (scale < 0) {
        scale = -scale;
        littleEndian = true;
    }

    // Allocate tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({height, width, channels}), &output));
    Tensor* output_scale = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &output_scale));

    if (data.size() != index + width * height * channels * sizeof(float)) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Invalid PFM data size, expected ", index + width * height * channels * sizeof(float)));
    }
    // Finish decoding image
    const uint8* innerData = (const uint8*)(data.data() + index);
    size_t length = height * width * channels;
    size_t size = length * sizeof(float);
    float* dstData = output->flat<float>().data();
    //std::memcpy(dstData, innerData, size);
    size_t row_size = width * channels * sizeof(float);
    // rows are swapped
    for (size_t i = 0; i < height; i++) {
        std::memcpy((uint8*)dstData + i * row_size, innerData + (height - i - 1) * row_size, row_size);
    }
    if (port::kLittleEndian != littleEndian) {
        uint8* bytes = (uint8*)dstData;
        for (size_t i = 0; i < size; i+=sizeof(float)) {
            for (size_t j = 0; j < sizeof(float) / 2; j++) {
                std::swap(bytes[i + j], bytes[i + (sizeof(float) - j - 1)]);
            }
        }
    }
    output_scale->scalar<float>()() = scale;
  }

 private:
  int channels_;
};
REGISTER_KERNEL_BUILDER(Name("DecodePfm").Device(DEVICE_CPU), DecodePfmOp);

