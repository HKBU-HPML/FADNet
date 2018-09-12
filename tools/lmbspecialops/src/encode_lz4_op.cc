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

REGISTER_OP("EncodeLz4")
    .Input("contents: dtype")
    .Attr("dtype: {float16, float32, float64, uint8, uint16, int8, int16, int32, int64} = DT_UINT8")
    .Output("result: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle unused;
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Encode LZ4-encoded data.
contents: n-D. The data.
result: 0-D. Encoded data.
)doc");


// Encode the contents of a LZ4 file
class EncodeLz4Op : public OpKernel {
 public:
  explicit EncodeLz4Op(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dt_));
    OP_REQUIRES(
      context, dt_ == DataType::DT_HALF || dt_ == DataType::DT_FLOAT || dt_ == DataType::DT_DOUBLE ||
      dt_ == DataType::DT_UINT8 || dt_ == DataType::DT_UINT16 ||
      dt_ == DataType::DT_INT8 || dt_ == DataType::DT_INT16 || dt_ == DataType::DT_INT32 || dt_ == DataType::DT_INT64,
      errors::InvalidArgument("Invalid dtype, got ", dt_));
  }

  template<typename T>
  void Compute_(OpKernelContext* context) {
    const Tensor& contents = context->input(0);

    const auto& flatContents = contents.flat<T>();

    // Create output
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));

    // Store output data
    std::string* output_data = &output->scalar<string>()();
    output_data->resize(LZ4_compressBound(flatContents.size() * sizeof(T)));
    int res = LZ4_compress_default((const char*)flatContents.data(), &(*output_data)[0], flatContents.size() * sizeof(T), output_data->size());
    OP_REQUIRES(context, res >= 0, errors::InvalidArgument("Encoding error: ", res));
    output_data->resize(res);
  }

  void Compute(OpKernelContext* context) override {
    switch(dt_) {
      case DataType::DT_HALF: Compute_<Eigen::half>(context); break;
      case DataType::DT_FLOAT: Compute_<float>(context); break;
      case DataType::DT_DOUBLE: Compute_<double>(context); break;
      case DataType::DT_UINT8: Compute_<uint8>(context); break;
      case DataType::DT_UINT16: Compute_<uint16>(context); break;
      case DataType::DT_INT8: Compute_<int8>(context); break;
      case DataType::DT_INT16: Compute_<int16>(context); break;
      case DataType::DT_INT32: Compute_<int32>(context); break;
      case DataType::DT_INT64: Compute_<int64>(context); break;
      default:
        OP_REQUIRES(context, false, errors::InvalidArgument("Invalid dtype: ", dt_));
    }
  }

private:
  DataType dt_;
};
REGISTER_KERNEL_BUILDER(Name("EncodeLz4").Device(DEVICE_CPU), EncodeLz4Op);

