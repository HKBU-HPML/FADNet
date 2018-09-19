#include "config.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("Correlation1D")
  .Attr("T: {float}")
  .Attr("corr_type: {'mult', 'subt'} = 'mult'")
  .Attr("max_displacement: int")
  .Attr("kernel_size: int")
  .Attr("stride1: int = 1")
  .Attr("stride2: int = 1")
  .Attr("pad_size: int = 0")
  .Attr("do_abs: bool = false")
  .Attr("single_dir: int = 0")
  .Input("input1: T")
  .Input("input2: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
      // check if both inputs have rank 4
      using namespace ::tensorflow::shape_inference;
      ShapeHandle input1_shape, input2_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input1_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &input2_shape));

      ShapeHandle merged_shape4d;
      TF_RETURN_IF_ERROR(c->Merge(input1_shape, input2_shape, &merged_shape4d));

      int kernel_size, max_displacement, pad_size, stride1, stride2, single_dir;

      // Get attributes
      TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));
      TF_RETURN_IF_ERROR(c->GetAttr("max_displacement", &max_displacement));
      TF_RETURN_IF_ERROR(c->GetAttr("single_dir", &single_dir));
      TF_RETURN_IF_ERROR(c->GetAttr("pad_size", &pad_size));
      TF_RETURN_IF_ERROR(c->GetAttr("stride1", &stride1));
      TF_RETURN_IF_ERROR(c->GetAttr("stride2", &stride2));

      int num      = c->Value(c->Dim(input1_shape, 0));
      int channels = c->Value(c->Dim(input1_shape, 1));
      int height   = c->Value(c->Dim(input1_shape, 2));
      int width    = c->Value(c->Dim(input1_shape, 3));

      int paddedbottomheight = height;
      int paddedbottomwidth = width+2*pad_size;

      // Size computation
      int kernel_radius = (kernel_size - 1) / 2; //size of unreachable border region (on each side)
      int border_size = max_displacement + kernel_radius; //size of unreachable border region (on each side)

      int top_width = ceil((float)(paddedbottomwidth - border_size*2) / (float)stride1);
      int top_height = ceil((float)(paddedbottomheight - kernel_radius*2) / (float)stride1);

      // Given a center position in image 1, how many displaced positions in -x / +x direction do we consider in image 2 (neighborhoodGridWidth):
      int neighborhood_grid_radius = max_displacement / stride2;
      int neighborhood_grid_width;

	  if(single_dir != 0) {
			neighborhood_grid_width = neighborhood_grid_radius + 1;
		} else {
			neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;
		}
      // Top Channels amount to displacement in X direction
      int top_channels = neighborhood_grid_width;

      // make sure warped has the same size as image
      c->set_output(0, c->MakeShape({num, top_channels, top_height, top_width}));

      return Status::OK();
    })
  .Doc(R"doc(
Correlates two feature maps in X direction.

Let K be the kernel size and N be the neighborhoud size.
For a K x K patch from the first image, (N-K+1) patches from the
second image are compared to it.

input1:
  Input tensor of rank 4 and data format "NCHW".

input2:
  Input tensor of rank 4 and data format "NCHW".

output:
  If W and H are the spatial input dimensions, this results in an output blob of
    For stride 1:
    Width: (W-K+1)
    Height: (H-K+1)
    Channels: N
)doc");

REGISTER_OP("Correlation1DGrad")
  .Attr("T: {float}")
  .Attr("corr_type: {'mult', 'subt'} = 'mult'")
  .Attr("max_displacement: int")
  .Attr("kernel_size: int")
  .Attr("stride1: int = 1")
  .Attr("stride2: int = 1")
  .Attr("pad_size: int = 0")
  .Attr("do_abs: bool = false")
  .Attr("single_dir: int = 0")
  .Input("input1: T")
  .Input("input2: T")
  .Input("gradient: T")
  .Output("input1_grad: T")
  .Output("input2_grad: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    })
  .Doc(R"doc(
This computes the gradient for the op 'Correlation1D'.
)doc");
