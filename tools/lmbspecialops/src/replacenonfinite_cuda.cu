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
#define EIGEN_USE_GPU
#include "config.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "helper.h"
#include "cuda_helper.h"

using namespace tensorflow;

namespace replacenonfinite_kernel_internal
{
  template <class T>
  __global__ void replacenonfinite_kernel(T* out, const T* in, const T value, const int size )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= size )
      return;

    T tmp = in[x];
    if( isfinite(tmp) )
      out[x] = tmp;
    else
      out[x] = value;
  }

  template <class T>
  __global__ void replacenonfinite_grad_kernel(T* out, const T* in, const T* grad, const int size )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= size )
      return;

    T tmp = in[x];
    if( isfinite(tmp) )
      out[x] = grad[x];
    else
      out[x] = T(0);
  }


} 
using namespace replacenonfinite_kernel_internal;


template <class T>
class ReplaceNonfiniteOp_GPU : public OpKernel 
{
public:
  explicit ReplaceNonfiniteOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
    float value_tmp;
    OP_REQUIRES_OK(construction, construction->GetAttr("value", &value_tmp));
    value = value_tmp;
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

    OP_REQUIRES(context, (size/256+1) < std::numeric_limits<int>::max(),
        errors::Internal("Input size is too large for ReplaceNonfinite on gpu device"));
    auto device = context->eigen_gpu_device();
    replacenonfinite_gpu(
        device.stream(),
        output.data(),
        input.data(),
        value,
        size);
    
  }

  void replacenonfinite_gpu( 
      const cudaStream_t& stream,
      T* out, const T* in, 
      const T value,
      const int64_t size )
  {
    dim3 block(256,1,1);
    dim3 grid;
    grid.x = divup(size,block.x);
    grid.y = 1;
    grid.z = 1;

    replacenonfinite_kernel<T><<<grid,block,0,stream>>>(out, in, value, size);
    CHECK_CUDA_ERROR;
  }


private:
  T value;
};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ReplaceNonfinite")                                                  \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ReplaceNonfiniteOp_GPU<type>);                                            
REG_KB(float)
REG_KB(double)
#undef REG_KB



template <class T>
class ReplaceNonfiniteGradOp_GPU : public OpKernel 
{
public:
  explicit ReplaceNonfiniteGradOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction)
  { }

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

    OP_REQUIRES(context, (size/256+1) < std::numeric_limits<int>::max(),
        errors::Internal("Input size is too large for ReplaceNonfiniteGrad on gpu device"));
    auto device = context->eigen_gpu_device();
    replacenonfinite_grad_gpu(
        device.stream(),
        output.data(),
        input.data(),
        gradients.data(),
        size);
  }


  void replacenonfinite_grad_gpu( 
      const cudaStream_t& stream,
      T* out, const T* in, const T* grad,
      const int64_t size )
  {
    dim3 block(256,1,1);
    dim3 grid;
    grid.x = divup(size,block.x);
    grid.y = 1;
    grid.z = 1;

    replacenonfinite_grad_kernel<T><<<grid,block,0,stream>>>(out, in, grad, size);
    CHECK_CUDA_ERROR;
  }


private:
  T value;
};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ReplaceNonfiniteGrad")                                              \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ReplaceNonfiniteGradOp_GPU<type>);                                        
REG_KB(float)
REG_KB(double)
#undef REG_KB

