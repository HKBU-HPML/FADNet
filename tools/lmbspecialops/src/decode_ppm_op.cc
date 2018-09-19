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

using namespace tensorflow;

REGISTER_OP("DecodePpm")
    .Input("contents: string")
    .Attr("dtype: {uint8, uint16} = DT_UINT8")
    .Output("image: dtype")
    .Output("maxval: dtype")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0,
                    c->MakeShape({InferenceContext::kUnknownDim,
                                  InferenceContext::kUnknownDim, 3}));
      c->set_output(1, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Decode a PPM-encoded image to a uint8 or uint16 tensor.
contents: 0-D.  The PPM-encoded image.
image: 3-D with shape `[height, width, 3]`.
maxval: maxval from the ppm file.
)doc");




// Decode the contents of a PPM file
class DecodePpmOp : public OpKernel {
 public:
  explicit DecodePpmOp(OpKernelConstruction* context) : OpKernel(context) {
    DataType dt;
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dt));
    OP_REQUIRES(
        context, dt == DataType::DT_UINT8 || dt == DataType::DT_UINT16,
        errors::InvalidArgument("Type must be UINT8 or UINT16, got ", dt));
    if (dt == DataType::DT_UINT8) {
      desired_channel_bits_ = 8;
    } else {
      desired_channel_bits_ = 16;
    }
  }

  static bool skipWhitespace(OpKernelContext* context, const StringPiece& data, size_t* index, bool onlyOne = false) {
      if ((*index) >= data.size()) {
        context->CtxFailure(errors::InvalidArgument("Invalid PPM header, unexpected end of file"));
        return false;
      }
      char c = data[(*index)];
      if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '#') {
        context->CtxFailure(errors::InvalidArgument("Invalid PPM header, expected whitespace at ", (*index)));
        return false;
      }
      (*index)++;
      if (onlyOne) {
        return true;
      }
      for (; (*index) < data.size(); (*index)++) {
          c = data[(*index)];
          // Skip comments
          if (c == '#') {
            for (; (*index) < data.size(); (*index)++) {
              c = data[(*index)];
              if (c == '\r' || c == '\n') {
                  break;
              }
            }
          }
          if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
              break;
          }
      }
      return true;
    }

    static bool readInt(OpKernelContext* context, const StringPiece& data, size_t* index, uint32* number) {
      if ((*index) >= data.size()) {
        context->CtxFailure(errors::InvalidArgument("Invalid PPM header, unexpected end of file"));
        return false;
      }
      char c = data[(*index)];
      if (c < '0' || c > '9') {
        context->CtxFailure(errors::InvalidArgument("Invalid PPM header, expected float at ", (*index)));
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

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));

    // Start decoding image to get shape details
    const StringPiece data = contents.scalar<string>()();
    if (data.size() < 9) {
      //P6<whitespace>0<whitespace>0<whitespace>0<whitespace><data>
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Invalid PPM data size, data too small for PPM file"));
    }
    if (!data.starts_with("P6")) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Invalid PPM header, expected 'P6'"));
    }

    size_t index = 2;
    if (!skipWhitespace(context, data, &index)) {
      return;
    }
    uint32 width = 0, height = 0, maxval = 0;
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
    if (!readInt(context, data, &index, &maxval)) {
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
                  errors::InvalidArgument("PPM size too large for int: ",
                                          width, " by ", height));
    }

    // Allocate tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({height, width, 3}), &output));
    Tensor* output_maxval = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &output_maxval));

    if (desired_channel_bits_ == 8 && maxval <= 255) {
      if (data.size() != index + width * height * 3 * sizeof(uint8)) {
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Invalid PPM data size, expected ", index + width * height * 3 * sizeof(uint8)));
      }
      // Finish decoding image
      const uint8* innerData = (const uint8*)(data.data() + index);
      uint8* dstData = output->flat<uint8>().data();
      std::memcpy(dstData, innerData, height * width * 3 * sizeof(uint8));
      output_maxval->scalar<uint8>()() = (uint8)maxval;
    } else if (desired_channel_bits_ == 16 && maxval > 255) {
      if (data.size() != index + width * height * 3 * sizeof(uint16)) {
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Invalid PPM data size, expected ", index + width * height * 3 * sizeof(uint16)));
      }
      // Finish decoding image
      const uint8* innerData = (const uint8*)(data.data() + index);
      uint16* dstData = output->flat<uint16>().data();
      std::memcpy(dstData, innerData, height * width * 3 * sizeof(uint16));
      // PPM data is always in big endian
      if (port::kLittleEndian) {
        // Change endianness from big endian to system endianness
        size_t size = height * width * 3 * sizeof(uint16);
        uint8* bytes = (uint8*)dstData;
        for (size_t i = 0; i < size; i+=sizeof(uint16)) {
          for (size_t j = 0; j < sizeof(uint16) / 2; j++) {
            std::swap(bytes[i + j], bytes[i + (sizeof(uint16) - j - 1)]);
          }
        }
      }

      output_maxval->scalar<uint16>()() = (uint16)maxval;
    } else {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("PPM maxval ", maxval, " does not match requested bit depth format ", desired_channel_bits_));
    }
  }

 private:
  int desired_channel_bits_;
};
REGISTER_KERNEL_BUILDER(Name("DecodePpm").Device(DEVICE_CPU), DecodePpmOp);

