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

#include <lz4.h>

using namespace tensorflow;

REGISTER_OP("DecodeLz4Raw")
    .Input("contents: string")
    .Attr("expected_size: int")
    .Output("result: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Decode LZ4-encoded data.
contents: 0-D. The LZ4-encoded data.
result: 0-D. Decoded data.
expected_size: Expected size of the output.
)doc");


// Decode the contents of a LZ4 file
class DecodeLz4RawOp : public OpKernel {
 public:
  explicit DecodeLz4RawOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("expected_size", &expected_size_));
    OP_REQUIRES(
              context,
              expected_size_ > 0,
              errors::InvalidArgument("expected size must be set", expected_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));

    // Start decoding
    const StringPiece data = contents.scalar<string>()();

    // Create output
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));

    // Store output data
    std::string* output_data = &output->scalar<string>()();
    output_data->resize(expected_size_);
    int res = LZ4_decompress_safe((const char*)data.data(), &(*output_data)[0], data.size(), output_data->size());
    OP_REQUIRES(context, res >= 0, errors::InvalidArgument("Invalid LZ4 data: ", res));
    output_data->resize(res);
  }

 private:
  int expected_size_;
};
REGISTER_KERNEL_BUILDER(Name("DecodeLz4Raw").Device(DEVICE_CPU), DecodeLz4RawOp);

