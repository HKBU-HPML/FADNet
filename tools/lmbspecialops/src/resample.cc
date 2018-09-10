#include "config.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("Resample")
  .Attr("T: {float, double}")
  .Attr("width: int")
  .Attr("height: int")
  .Attr("antialias: bool=True")
  .Attr("type: {'NEAREST', 'CUBIC', 'LINEAR'} = 'LINEAR'")
  .Input("input:T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
      using namespace ::tensorflow::shape_inference;
	  int width, height;
	  c->GetAttr("width", &width);
	  c->GetAttr("height", &height);
	  c->set_output(0, c->MakeShape({ c->Dim(c->input(0), 0),
						              c->Dim(c->input(0), 1),
											 height,
											 width
										  }
										  ));
      return Status::OK();
    })
  .Doc(R"doc(
  Resamples an input with a specified width and height
  input:
  	the input tensor to resample
  type:
    the resample type CUBIC, LINEAR or NEAREST
  width:
	width of the output
  height:
    height of the output
  antialias:
    bool flag to choose if antialiasing is needed or not.
    Default value is True.
  output:
  	the output containing resampled output
)doc");


template <class T>
class ResampleOp: public OpKernel
{
	int width;
	int height;
public:
  explicit ResampleOp(OpKernelConstruction* c)
    :OpKernel(c)
  {
	OP_REQUIRES_OK(c,c->GetAttr("width", &width));
	OP_REQUIRES_OK(c,c->GetAttr("height", &height));
  }

  void Compute( OpKernelContext* context ) override
  {
	  //throw an error here
  }
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("Resample")                                                         \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ResampleOp<type>);
REG_KB(float)
REG_KB(double)
#undef REG_KB
