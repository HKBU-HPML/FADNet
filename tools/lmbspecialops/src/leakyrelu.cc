//
//  lmbspecialops - a collection of tensorflow ops
//  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
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

using namespace tensorflow;

REGISTER_OP("LeakyRelu")
  .Attr("T: {float, double}")
  .Attr("leak: float = 0.1")
  .Input("input: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
  .Doc(R"doc(
Computes the leaky rectified linear unit activations y = max(leak*x,x).

leak:
  The leak factor. 

input: 
  Input tensor of any shape.

output:
  A tensor with the activation.
)doc");



template <class T>
class LeakyReluOp : public OpKernel 
{
public:
  explicit LeakyReluOp(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
    float leak_tmp;
    OP_REQUIRES_OK(construction, construction->GetAttr("leak", &leak_tmp));
    leak = leak_tmp;
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());

    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    const int64_t size = input_shape.num_elements();

    const T* in_ptr = input.data();
    T* out_ptr = output.data();
    for( int64_t i = 0; i < size; ++i )
    {
      const T tmp = in_ptr[i];
      out_ptr[i] = std::max(leak*tmp,tmp);
    }
    
  }

private:
  T leak;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("LeakyRelu")                                                         \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    LeakyReluOp<type>);                                                       
REG_KB(float)
REG_KB(double)
#undef REG_KB




REGISTER_OP("LeakyReluGrad")
  .Attr("T: {float, double}")
  .Attr("leak: float")
  .Input("gradients: T")
  .Input("input: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
  .Doc(R"doc(
This computes the gradient for the op 'LeakyRelu'. 
)doc");


template <class T>
class LeakyReluGradOp : public OpKernel 
{
public:
  explicit LeakyReluGradOp(OpKernelConstruction* construction)
    :OpKernel(construction)
  {
    float leak_tmp;
    OP_REQUIRES_OK(construction, construction->GetAttr("leak", &leak_tmp));
    leak = leak_tmp;
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& gradients_tensor = context->input(0);
    auto gradients = gradients_tensor.flat<T>();
    const TensorShape gradients_shape(gradients_tensor.shape());
    const Tensor& input_tensor = context->input(1);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());

    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    const int64_t size = input_shape.num_elements();

    const T* in_ptr = input.data();
    const T* grad_ptr = gradients.data();
    T* out_ptr = output.data();
    for( int64_t i = 0; i < size; ++i )
    {
      const T tmp = in_ptr[i];
      const T leak_tmp = leak*tmp;
      if( tmp >= leak_tmp )
        out_ptr[i] = grad_ptr[i];
      else
        out_ptr[i] = leak*grad_ptr[i];
    }
    
  }

private:
  T leak;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("LeakyReluGrad")                                                     \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    LeakyReluGradOp<type>);                                                   
REG_KB(float)
REG_KB(double)
#undef REG_KB


