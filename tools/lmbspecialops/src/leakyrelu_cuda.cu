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

namespace leakyrelu_kernel_internal
{

  __device__ __forceinline__ float Max(const float a, const float b)
  {
    return fmaxf(a,b);
  }
  __device__ __forceinline__ double Max(const double a, const double b)
  {
    return fmax(a,b);
  }

  template <class T>
  __global__ void leakyrelu_kernel(T* out, const T* in, const T leak, const int size )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= size )
      return;

    const T tmp = in[x];
    out[x] = Max(leak*tmp,tmp);
  }

  template <class T>
  __global__ void leakyrelu_grad_kernel(T* out, const T* in, const T* grad, const T leak, const int size )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= size )
      return;

    const T tmp = in[x];
    const T leak_tmp = leak*tmp;
    if( tmp >= leak_tmp )
      out[x] = grad[x];
    else
      out[x] = leak*grad[x];
  }


} 
using namespace leakyrelu_kernel_internal;




template <class T>
class LeakyReluOp_GPU : public OpKernel 
{
public:
  explicit LeakyReluOp_GPU(OpKernelConstruction* construction)
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

    OP_REQUIRES(context, (size/256+1) < std::numeric_limits<int>::max(),
        errors::Internal("Input size is too large for LeakyReluOp on gpu device"));
    auto device = context->eigen_gpu_device();
    leakyrelu_gpu(
        device.stream(),
        output.data(),
        input.data(),
        leak,
        size);
    
  }


  void leakyrelu_gpu( 
      const cudaStream_t& stream,
      T* out, const T* in, 
      const T leak,
      const int64_t size )
  {
    dim3 block(256,1,1);
    dim3 grid;
    grid.x = divup(size,block.x);
    grid.y = 1;
    grid.z = 1;

    leakyrelu_kernel<T><<<grid,block,0,stream>>>(out, in, leak, size);
    CHECK_CUDA_ERROR;
  }

private:
  T leak;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("LeakyRelu")                                                         \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    LeakyReluOp_GPU<type>);                                                   
REG_KB(float)
REG_KB(double)
#undef REG_KB


template <class T>
class LeakyReluGradOp_GPU : public OpKernel 
{
public:
  explicit LeakyReluGradOp_GPU(OpKernelConstruction* construction)
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

    OP_REQUIRES(context, (size/256+1) < std::numeric_limits<int>::max(),
        errors::Internal("Input size is too large for LeakyReluGrad on gpu device"));
    auto device = context->eigen_gpu_device();
    leakyrelu_grad_gpu(
        device.stream(),
        output.data(),
        input.data(),
        gradients.data(),
        leak,
        size);
    
  }

  void leakyrelu_grad_gpu( 
      const cudaStream_t& stream,
      T* out, const T* in, const T* grad, const T leak,
      const int64_t size )
  {

    dim3 block(256,1,1);
    dim3 grid;
    grid.x = divup(size,block.x);
    grid.y = 1;
    grid.z = 1;

    leakyrelu_grad_kernel<T><<<grid,block,0,stream>>>(out, in, grad, leak, size);
    CHECK_CUDA_ERROR;
  }


private:
  T leak;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("LeakyReluGrad")                                                     \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    LeakyReluGradOp_GPU<type>);                                               
REG_KB(float)
REG_KB(double)
#undef REG_KB

