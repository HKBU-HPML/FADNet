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

REGISTER_OP("EncodePfm")
    .Input("image: float")
    .Attr("scale: float = -1.0")
    .Output("contents: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Encode float flow data as PFM file.
image: 3-D with shape `[height, width, {1|3}]`.
scale: Scalar containing the scale. Negative for little endian encoding.
contents: 0-D.  The PFM-encoded data.
)doc");




// Encode the contents to a PFM file
class EncodePfmOp : public OpKernel {
 public:
  explicit EncodePfmOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("data must be 3-dimensional",
                                        image.shape().DebugString()));
    const uint32 height = static_cast<uint32>(image.dim_size(0));
    const uint32 width = static_cast<uint32>(image.dim_size(1));
    const uint32 channels = static_cast<uint32>(image.dim_size(2));

    OP_REQUIRES(context, channels == 1 || channels == 3,
                        errors::InvalidArgument("data must be [height, width, {1|3}]-shaped",
                                                image.shape().DebugString()));

    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));

    std::string* output_data = &output->scalar<string>()();
    if (channels == 1) {
        output_data->append("Pf");
    } else {
        output_data->append("PF");
    }
    output_data->append("\n");
    output_data->append(std::to_string(width));
    output_data->append(" ");
    output_data->append(std::to_string(height));
    output_data->append("\n");
    output_data->append(std::to_string(scale_));
    output_data->append("\n");
    uint8* innerData = (uint8*)image.flat<float>().data();

    size_t offset = output_data->size();
    size_t length = height * width * channels;
    size_t size = length * sizeof(float);
    size_t row_size = width * channels * sizeof(float);

    output_data->reserve(offset + size);

    bool littleEndian = (scale_ < 0.0f);
    // rows are swapped
    for (size_t i = 0; i < height; i++) {
        output_data->append((const char*)(innerData + (height - i - 1) * row_size), row_size);
    }
    // Swap endianness
    if (port::kLittleEndian != littleEndian) {
        for (size_t i = 0; i < size; i+=sizeof(float)) {
            std::swap((*output_data)[offset + i + 0], (*output_data)[offset + i + 3]);
            std::swap((*output_data)[offset + i + 1], (*output_data)[offset + i + 2]);
        }
    }
  }

private:
  float scale_;
};
REGISTER_KERNEL_BUILDER(Name("EncodePfm").Device(DEVICE_CPU), EncodePfmOp);

