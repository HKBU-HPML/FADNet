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

REGISTER_OP("DecodeFlo")
    .Input("contents: string")
    .Output("image: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;
      c->set_output(0,
                    c->MakeShape({InferenceContext::kUnknownDim,
                                  InferenceContext::kUnknownDim, 2}));
      return Status::OK();
    })
    .Doc(R"doc(
Decode a FLO-encoded image to a float tensor.
contents: 0-D.  The FLO-encoded image.
image: 3-D with shape `[height, width, 2]`.
)doc");




// Decode the contents of a FLO file
class DecodeFloOp : public OpKernel {
 public:
  explicit DecodeFloOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));

    // Start decoding image to get shape details
    const StringPiece data = contents.scalar<string>()();
    if (data.size() < 12) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Invalid FLO data size, expected at least 12"));
    }

    if (!data.starts_with("PIEH")) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Invalid FLO header, expected 'PIEH'"));
    }
    if (*((float*)(data.data())) != 202021.25f) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Invalid FLO header, expected 202021.25 (sanity check failed)"));
    }
    uint32 width = *((uint32*)(data.data() + 4));
    uint32 height = *((uint32*)(data.data() + 8));

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
                  errors::InvalidArgument("FLO size too large for int: ",
                                          width, " by ", height));
    }

    if (data.size() != 12 + width * height * 2 * 4) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Invalid FLO data size, expected ", 12 + width * height * 2 * 4));
    }

    // Allocate tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({height, width, 2}), &output));

    // Finish decoding image
    const uint8* innerData = (const uint8*)(data.data() + 12);
    memcpy(output->flat<float>().data(), innerData, height * width * 2 * sizeof(float));
  }
};
REGISTER_KERNEL_BUILDER(Name("DecodeFlo").Device(DEVICE_CPU), DecodeFloOp);

