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

REGISTER_OP("EncodeLz4Raw")
    .Input("contents: string")
    .Output("result: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;
        using namespace ::tensorflow::shape_inference;
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
        c->set_output(0, c->Scalar());
        return Status::OK();
    })
    .Doc(R"doc(
Encode LZ4-encoded data.
contents: 0-D. The data.
result: 0-D. Encoded data.
)doc");


// Encode the contents of a LZ4 file
class EncodeLz4RawOp : public OpKernel {
 public:
  explicit EncodeLz4RawOp(OpKernelConstruction* context) : OpKernel(context) {
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
    output_data->resize(LZ4_compressBound(data.size()));
    int res = LZ4_compress_default((const char*)data.data(), &(*output_data)[0], data.size(), output_data->size());
    OP_REQUIRES(context, res >= 0, errors::InvalidArgument("Invalid LZ4 data: ", res));
    output_data->resize(res);
  }
};
REGISTER_KERNEL_BUILDER(Name("EncodeLz4Raw").Device(DEVICE_CPU), EncodeLz4RawOp);

