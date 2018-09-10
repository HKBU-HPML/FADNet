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

REGISTER_OP("EncodeFlo")
    .Input("image: float")
    .Output("contents: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Encode float flow data as FLO file.
image: 3-D with shape `[height, width, 2]`.
contents: 0-D.  The FLO-encoded data.
)doc");




// Encode the contents to a FLO file
class EncodeFloOp : public OpKernel {
 public:
  explicit EncodeFloOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("data must be 3-dimensional",
                                        image.shape().DebugString()));
    OP_REQUIRES(context, image.dim_size(2) == 2,
                    errors::InvalidArgument("data must be [height, width, 2]-shaped",
                                            image.shape().DebugString()));

    const int32 height = static_cast<int32>(image.dim_size(0));
    const int32 width = static_cast<int32>(image.dim_size(1));

    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));

    std::string* output_data = &output->scalar<string>()();
    output_data->reserve(12 + width * height * 2 * sizeof(float));
    output_data->append("PIEH", 4);
    output_data->append((const char*)&width, sizeof(int32));
    output_data->append((const char*)&height, sizeof(int32));
    output_data->append((const char*)image.flat<float>().data(), width * height * 2 * sizeof(float));
  }
};
REGISTER_KERNEL_BUILDER(Name("EncodeFlo").Device(DEVICE_CPU), EncodeFloOp);

