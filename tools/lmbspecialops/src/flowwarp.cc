//
//  lmbspecialops - a collection of tensorflow ops
//  Copyright (C) 2017  Oezguen Cicek, Eddy Ilg
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
#include <math.h>

#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))

using namespace tensorflow;

REGISTER_OP("FlowWarp")
  .Attr("T: {float}")
  .Attr("fill_parameter: {'zero', 'not_a_number'} = 'zero'")
  .Input("image: T")
  .Input("flow: T")
  .Output("warped: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    { 
      // check if both inputs have rank 4
      using namespace ::tensorflow::shape_inference;
      ShapeHandle image_shape, flow_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &image_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &flow_shape));

      // check if width and height are compatible
      ShapeHandle image_shape2d;
      c->Subshape(image_shape, -2, &image_shape2d);
      ShapeHandle flow_shape2d;
      c->Subshape(flow_shape, -2, &flow_shape2d);

      ShapeHandle merged_shape2d;
      TF_RETURN_IF_ERROR(c->Merge(image_shape2d, flow_shape2d, &merged_shape2d));
      
      // check if N is compatible for both inputs
      DimensionHandle image_first_dim = c->Dim(image_shape, 0);
      DimensionHandle flow_first_dim  = c->Dim(flow_shape,  0);

      DimensionHandle d_first;
      TF_RETURN_IF_ERROR(c->Merge(image_first_dim, flow_first_dim, &d_first));
      
      // check if C == 2 for the flow
      DimensionHandle d_c;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(1), 1), 2, &d_c));
        
      // make sure warped has the same size as image
      c->set_output(0, c->input(0));
      return Status::OK();
    })
  .Doc(R"doc(
Warps the image with the given flow.

image: 
  Input tensor of rank 4 and data format "NCHW".
  
flow: 
  Input tensor of rank 4 and data format "NCHW".

warped:
  A tensor of rank 4 and data format "NCHW" for the warped image.
)doc");
  
template<class T>
class FlowWarpOp : public OpKernel 
{
public:
  explicit FlowWarpOp(OpKernelConstruction* construction) 
    :OpKernel(construction) 
  {
    std::string fill_parameter_str;
    OP_REQUIRES_OK(construction, construction->GetAttr("fill_parameter", &fill_parameter_str));
    if( fill_parameter_str == "zero" )
      fill_parameter = ZERO;
    else 
      fill_parameter = NOT_A_NUMBER;
  }

  void Compute(OpKernelContext* context) override 
  {
    // Get the inputs   
    const Tensor& image_tensor = context->input(0);
    const Tensor& flow_tensor  = context->input(1);
    
    // Get the shapes
    const TensorShape image_shape(image_tensor.shape());
    
    // Allocate the memory for the warped image
    Tensor* warped_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, image_shape, &warped_tensor));

    // Prepare for warping
    auto image  = image_tensor.flat<T>();
    auto flow   = flow_tensor.flat<T>();
    auto warped = warped_tensor->flat<T>();
     
    int num      = image_shape.dim_size(0);
    int channels = image_shape.dim_size(1);
    int height   = image_shape.dim_size(2);
    int width    = image_shape.dim_size(3);
    
    FlowWarp_CPU(warped.data(), image.data(), flow.data(), num, channels, height, width);
  }
    
  void FlowWarp_CPU( T*           warped,
                     const T*     image,
                     const T*     flow,
                     const int    num,
                     const int    channels,
                     const int    height,
                     const int    width
                   ) 
  {
    const int wh_size = width * height;
    const int whc_size = width * height * channels;
      
    float fill_value = fill_parameter == ZERO ? 0 : NAN;
    
    for(int n=0; n<num; n++)
    {
        int off = whc_size * n;
        for(int x=0; x<width; x++)
            for(int y=0; y<height; y++)
            {
                float fx = flow[2*wh_size*n + y*width + x];
                float fy = flow[2*wh_size*n + wh_size + y*width + x];

                float x2 = float(x) + fx;
                float y2 = float(y) + fy;

                if(x2>=0 && y2>=0 && x2<width && y2<height)
                {
                    int ix2_L = int(x2);
                    int iy2_T = int(y2);
                    int ix2_R = min(ix2_L+1, width-1);
                    int iy2_B = min(iy2_T+1, height-1);

                    float alpha=x2-ix2_L;
                    float beta=y2-iy2_T;

                    for(int c=0; c<channels; c++)
                    {
                        float TL = image[off + c*wh_size + iy2_T*width + ix2_L];
                        float TR = image[off + c*wh_size + iy2_T*width + ix2_R];
                        float BL = image[off + c*wh_size + iy2_B*width + ix2_L];
                        float BR = image[off + c*wh_size + iy2_B*width + ix2_R];

                        warped[off + c*wh_size + y*width + x] =
                            (1-alpha)*(1-beta)*TL +
                            alpha*(1-beta)*TR +
                            (1-alpha)*beta*BL +
                            alpha*beta*BR;
                    }
                }
                else
                {
                    for(int c=0; c<channels; c++)
                        warped[off + c*wh_size + y*width + x] = fill_value;
                }
            }
    }
  }
private:
    enum FillParameter {ZERO = 1, NOT_A_NUMBER = 2};
    FillParameter fill_parameter;
};
  
#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("FlowWarp")                                                          \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                         \
    FlowWarpOp<type>);                                                    
REG_KB(float)
#undef REG_KB

REGISTER_OP("FlowWarpGrad")
  .Attr("T: {float}")
  .Input("image: T")
  .Input("flow: T")
  .Input("gradient: T")
  .Output("image_grad: T")
  .Output("flow_grad: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    })
  .Doc(R"doc(
This computes the gradient for the op 'FlowWarp'. 
)doc");
  
template <class T>
class FlowWarpGradOp : public OpKernel 
{
public:
  explicit FlowWarpGradOp(OpKernelConstruction* construction)
    :OpKernel(construction) {}
    
  void Compute( OpKernelContext* context ) override 
  {
    // Get the inputs   
    const Tensor& image_tensor     = context->input(0);
    const Tensor& flow_tensor      = context->input(1); 
    const Tensor& gradient_tensor  = context->input(2);
    
    // Get the shapes
    const TensorShape image_shape(image_tensor.shape());
    const TensorShape flow_shape(flow_tensor.shape());
    
    // Allocate the memory for the output
    Tensor* image_grad_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, image_shape, &image_grad_tensor));
    Tensor* flow_grad_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, flow_shape,  &flow_grad_tensor));
    
    // Prepare for gradient computation    
    auto flow        = flow_tensor.flat<T>(); 
    auto image       = image_tensor.flat<T>();
    auto gradient    = gradient_tensor.flat<T>();
    auto image_grad  = image_grad_tensor->flat<T>();
    auto flow_grad   = flow_grad_tensor->flat<T>();

    int num      = image_shape.dim_size(0);
    int channels = image_shape.dim_size(1);
    int height   = image_shape.dim_size(2);
    int width    = image_shape.dim_size(3);
    
    FlowWarpGrad_CPU(image_grad.data(), flow_grad.data(), image.data(), flow.data(), gradient.data(), num, channels, height, width);
  }
 
  void FlowWarpGrad_CPU( T*           image_grad,
                         T*           flow_grad,
                         const T*     image,
                         const T*     flow,
                         const T*     gradient,
                         const int    num,
                         const int    channels,
                         const int    height,
                         const int    width
                       ) 
  {
    const int wh_size = width * height;
    const int whc_size = width * height * channels; 
    
    for(int i=0; i<num*whc_size; i++)
        image_grad[i] = 0 ;
    for(int i=0; i<num*whc_size; i++)
        flow_grad[i] = 0 ;
    
    for(int n=0; n<num; n++)
    {
        int off = whc_size * n;
        for(int x=0; x<width; x++)
            for(int y=0; y<height; y++)
            {
                float fx = flow[2*wh_size*n + y*width + x];
                float fy = flow[2*wh_size*n + wh_size + y*width + x];

                float x2 = float(x) + fx;
                float y2 = float(y) + fy;

                if(x2>=0 && y2>=0 && x2<width && y2<height)
                {
                    int ix2_L = int(x2);
                    int iy2_T = int(y2);
                    int ix2_R = min(ix2_L+1, width-1);
                    int iy2_B = min(iy2_T+1, height-1);

                    float alpha=x2-ix2_L;
                    float beta=y2-iy2_T;
                    for(int c=0; c<channels; c++)
                    {
                        float warped_diff_value = gradient[off + c*wh_size + y*width + x];
                        image_grad[off + c*wh_size + iy2_T*width + ix2_L] += warped_diff_value * (1-alpha)*(1-beta);
                        image_grad[off + c*wh_size + iy2_T*width + ix2_R] += warped_diff_value * alpha*(1-beta);
                        image_grad[off + c*wh_size + iy2_B*width + ix2_L] += warped_diff_value * (1-alpha)*beta;
                        image_grad[off + c*wh_size + iy2_B*width + ix2_R] += warped_diff_value * alpha*beta;
                    }

                    float gamma = iy2_B - y2;
                    float bot_diff = 0;
                    for(int c=0; c<channels; c++)
                    {
                        float temp = 0;
                        temp += gamma *     (image[off + c*wh_size + iy2_T*width + ix2_R] - image[off + c*wh_size + iy2_T*width + ix2_L]);
                        temp += (1-gamma) * (image[off + c*wh_size + iy2_B*width + ix2_R] - image[off + c*wh_size + iy2_B*width + ix2_L]);

                        bot_diff += gradient[off + c*wh_size + y*width + x] * temp;
                    }
                    flow_grad[2*wh_size*n + y*width + x] = bot_diff;

                    gamma = ix2_R - x2;
                    bot_diff = 0;
                    for(int c=0; c<channels; c++)
                    {
                        float temp = 0;
                        temp += gamma *     (image[off + c*wh_size + iy2_B*width + ix2_L] - image[off + c*wh_size + iy2_T*width + ix2_L]);
                        temp += (1-gamma) * (image[off + c*wh_size + iy2_B*width + ix2_R] - image[off + c*wh_size + iy2_T*width + ix2_R]);

                        bot_diff += gradient[off + c*wh_size + y*width + x] * temp;
                    }
                    flow_grad[2*wh_size*n + wh_size + y*width + x] = bot_diff;
                }
            }
    }
  }
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("FlowWarpGrad")                                                      \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    FlowWarpGradOp<type>);                                                   
REG_KB(float)
#undef REG_KB
