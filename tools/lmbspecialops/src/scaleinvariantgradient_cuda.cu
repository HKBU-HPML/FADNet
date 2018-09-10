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

namespace scaleinvariantgrad_kernel_internal
{

  template <class T>
  __device__ __forceinline__ bool is_valid(const T& value)
  {
    return isfinite(value);
  }

  template <class T>
  __device__ __forceinline__ T dcenter(const T& center_value, const T& neighbour_value, const T& eps)
  {
    T sum_abs = std::abs(center_value) + std::abs(neighbour_value) + eps;
    T sign = -1;
    if( center_value < 0 )
      sign = 1;
    return -1/sum_abs + sign*(neighbour_value - center_value)/(sum_abs*sum_abs);
  }

  template <class T>
  __device__ __forceinline__ T dneighbour(const T& center_value, const T& neighbour_value, const T& eps)
  {
    T sum_abs = std::abs(center_value) + std::abs(neighbour_value) + eps;
    T sign = -1;
    if( neighbour_value < 0 )
      sign = 1;
    return 1/sum_abs + sign*(neighbour_value - center_value)/(sum_abs*sum_abs);
  }


  template <class T>
  __global__ void computeForward( 
      T* out, const T* input,
      int x_size, int y_size, int z_size, T eps,
      const int* deltas, const T* weights, int max_comparisons )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if( x >= x_size || y >= y_size || z >= z_size )
      return;

    const int xy_size = x_size*y_size;

#define INPUT(x,y,z) input[(z)*xy_size+(y)*x_size+(x)]
#define OUT(x,y,z) out[(z)*xy_size+(y)*x_size+(x)]

    const T value0 = INPUT(x,y,z);
    T grad_x = 0;
    T grad_y = 0;

    for( int comparison = 0; comparison < max_comparisons; ++comparison )
    {
      int delta = deltas[comparison];
      T weight = weights[comparison];

      T valuex, valuey;
      if( x+delta >= 0 && x+delta < x_size )
        valuex = INPUT(x+delta,y,z);
      else
        valuex = value0;

      if( y+delta >= 0 && y+delta < y_size )
        valuey = INPUT(x,y+delta,z);
      else
        valuey = value0;

      grad_x += weight*(valuex-value0)/(std::abs(value0)+std::abs(valuex)+eps);
      grad_y += weight*(valuey-value0)/(std::abs(value0)+std::abs(valuey)+eps);
    }

    OUT(x,y,2*z+0) = grad_x;
    OUT(x,y,2*z+1) = grad_y;

#undef INPUT
#undef OUT
  }




  template <class T>
  __global__ void computeBackward( 
      T* out, const T* input_data, const T* grad, 
      int x_size, int y_size, int z_size, T eps,
      const int* deltas, const T* weights, int max_comparisons )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if( x >= x_size || y >= y_size || z >= z_size )
      return;

    const int xy_size = x_size*y_size;

#define INPUT(x,y,z) input_data[(z)*xy_size+(y)*x_size+(x)]
#define OUT(x,y,z) out[(z)*xy_size+(y)*x_size+(x)]
#define GRAD(x,y,z) grad[(z)*xy_size+(y)*x_size+(x)]

    T value0_diff = 0;
    const T value0 = INPUT(x,y,z);

    if( is_valid(value0) )
    {
      for( int comparison = 0; comparison < max_comparisons; ++comparison )
      {
        int delta = deltas[comparison];
        T weight = weights[comparison];

        T value0_diff_tmp = 0;

          // compute the backpropagated gradient from the x component
          if( x+delta >= 0 && x+delta < x_size )
          {
            const T valuex = INPUT(x+delta,y,z);
            if( is_valid(valuex) )
            {
              T tmp = dcenter(value0, valuex, eps);
              value0_diff_tmp += tmp * GRAD(x,y,2*z+0);
            }
          }
          if( x-delta >= 0 && x-delta < x_size )
          {
            const T valuex = INPUT(x-delta,y,z);
            if( is_valid(valuex) )
            {
              T tmp = dneighbour(valuex, value0, eps);
              value0_diff_tmp += tmp * GRAD(x-delta,y,2*z+0);
            }
          } 

          // compute the backpropagated gradient from the y component
          if( y+delta >= 0 && y+delta < y_size )
          {
            const T valuey = INPUT(x,y+delta,z);
            if( is_valid(valuey) )
            {
              T tmp = dcenter(value0, valuey, eps);
              value0_diff_tmp += tmp * GRAD(x,y,2*z+1);
            }
          }
          if( y-delta >= 0 && y-delta < y_size )
          {
            const T valuey = INPUT(x,y-delta,z);
            if( is_valid(valuey) )
            {
              T tmp = dneighbour(valuey, value0, eps);
              value0_diff_tmp += tmp * GRAD(x,y-delta,2*z+1);
            }
          } 
          value0_diff += weight*value0_diff_tmp;
       }
    }

    if( !isfinite(value0_diff) )
      value0_diff = 0;
    OUT(x,y,z) = value0_diff;

#undef INPUT
#undef OUT
#undef GRAD

  }

} 
using namespace scaleinvariantgrad_kernel_internal;




template <class T>
class ScaleInvariantGradientOp_GPU : public OpKernel 
{
public:
  explicit ScaleInvariantGradientOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction), persistent_initialized(false)
  { 
    std::vector<float> weights_tmp;
    float eps_tmp;
    OP_REQUIRES_OK(construction, construction->GetAttr("deltas", &deltas));
    OP_REQUIRES_OK(construction, construction->GetAttr("weights", &weights_tmp));
    OP_REQUIRES_OK(construction, construction->GetAttr("epsilon", &eps_tmp));
    epsilon = eps_tmp;
    for( float w : weights_tmp )
      weights.push_back(w);

    OP_REQUIRES(construction, deltas.size() == weights.size(),
        errors::InvalidArgument("The size of the deltas and weights vectors must be the same")
        );

  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());
    const int input_rank = input_shape.dims();

    TensorShape output_shape;
    int64_t w_size = 1;
    for( int i = 0; i < input_rank-2; ++i )
      w_size *= input_shape.dim_size(i);
    output_shape.AddDim(w_size);
    output_shape.AddDim(2);
    output_shape.AddDim(input_shape.dim_size(input_rank-2));
    output_shape.AddDim(input_shape.dim_size(input_rank-1));
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    auto device = context->eigen_gpu_device();
    Tensor* deltas_gpu_tensor;
    Tensor* weights_gpu_tensor;

    if( !persistent_initialized )
    {

      OP_REQUIRES_OK(context, context->allocate_persistent(
            DataTypeToEnum<int>::v(),
            TensorShape({int64_t(deltas.size())}),
            &deltas_gpu,
            &deltas_gpu_tensor ));
      int* deltas_gpu_ptr = deltas_gpu_tensor->flat<int>().data();

      //device.memcpyHostToDevice(deltas_gpu_ptr, &deltas[0], sizeof(T)*deltas.size());
      cudaMemcpyAsync(deltas_gpu_ptr, &deltas[0], sizeof(T)*deltas.size(), cudaMemcpyHostToDevice, device.stream());


      OP_REQUIRES_OK(context, context->allocate_persistent(
            DataTypeToEnum<T>::v(),
            TensorShape({int64_t(weights.size())}),
            &weights_gpu,
            &weights_gpu_tensor ));
      T* weights_gpu_ptr = weights_gpu_tensor->flat<T>().data();

      //device.memcpyHostToDevice(weights_gpu_ptr, &weights[0], sizeof(T)*weights.size());
      cudaMemcpyAsync(weights_gpu_ptr, &weights[0], sizeof(T)*weights.size(), cudaMemcpyHostToDevice, device.stream());
      persistent_initialized = true;
    }

    deltas_gpu_tensor = deltas_gpu.AccessTensor(context);
    weights_gpu_tensor = weights_gpu.AccessTensor(context);

    scaleinvariantgrad_gpu( 
        device.stream(),
        output.data(),
        input.data(),
        deltas_gpu_tensor->flat<int>().data(),
        weights_gpu_tensor->flat<T>().data(),
        deltas.size(),
        epsilon,
        input_shape.dim_size(input_rank-1),
        input_shape.dim_size(input_rank-2),
        w_size );
    
  }


  void scaleinvariantgrad_gpu( 
      const cudaStream_t& stream,
      T* out, const T* in, 
      const int* deltas,
      const T* weights,
      int num_deltas,
      T epsilon,
      int x_size, int y_size, int z_size )
  {
    dim3 block(32,4,1);
    dim3 grid;
    grid.x = divup(x_size,block.x);
    grid.y = divup(y_size,block.y);
    grid.z = divup(z_size,block.z);

    computeForward<T><<<grid,block,0,stream>>>(
        out, in,
        x_size, y_size, z_size, 
        epsilon,
        deltas, weights, num_deltas );
    CHECK_CUDA_ERROR
  }


private:
  std::vector<int> deltas;
  std::vector<T> weights;
  T epsilon;
  PersistentTensor deltas_gpu;
  PersistentTensor weights_gpu;
  bool persistent_initialized;

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ScaleInvariantGradient")                                            \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ScaleInvariantGradientOp_GPU<type>);                                      
REG_KB(float)
REG_KB(double)
#undef REG_KB




template <class T>
class ScaleInvariantGradientGradOp_GPU : public OpKernel 
{
public:
  explicit ScaleInvariantGradientGradOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction), persistent_initialized(false)
  { 
    std::vector<float> weights_tmp;
    float eps_tmp;
    OP_REQUIRES_OK(construction, construction->GetAttr("deltas", &deltas));
    OP_REQUIRES_OK(construction, construction->GetAttr("weights", &weights_tmp));
    OP_REQUIRES_OK(construction, construction->GetAttr("epsilon", &eps_tmp));
    epsilon = eps_tmp;
    for( float w : weights_tmp )
      weights.push_back(w);

    OP_REQUIRES(construction, deltas.size() == weights.size(),
        errors::InvalidArgument("The size of the deltas and weights vectors must be the same")
        );

  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& gradient_tensor = context->input(0);
    auto gradient = gradient_tensor.flat<T>();
    const TensorShape gradient_shape(gradient_tensor.shape());
    
    const Tensor& input_tensor = context->input(1);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());
    const int input_rank = input_shape.dims();

    TensorShape output_shape(input_shape);
    int64_t w_size = 1;
    for( int i = 0; i < input_rank-2; ++i )
      w_size *= input_shape.dim_size(i);
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    auto device = context->eigen_gpu_device();
    Tensor* deltas_gpu_tensor;
    Tensor* weights_gpu_tensor;

    if( !persistent_initialized )
    {

      OP_REQUIRES_OK(context, context->allocate_persistent(
            DataTypeToEnum<int>::v(),
            TensorShape({int64_t(deltas.size())}),
            &deltas_gpu,
            &deltas_gpu_tensor ));
      int* deltas_gpu_ptr = deltas_gpu_tensor->flat<int>().data();

      cudaMemcpyAsync(deltas_gpu_ptr, &deltas[0], sizeof(T)*deltas.size(), cudaMemcpyHostToDevice, device.stream());

      OP_REQUIRES_OK(context, context->allocate_persistent(
            DataTypeToEnum<T>::v(),
            TensorShape({int64_t(weights.size())}),
            &weights_gpu,
            &weights_gpu_tensor ));
      T* weights_gpu_ptr = weights_gpu_tensor->flat<T>().data();

      cudaMemcpyAsync(weights_gpu_ptr, &weights[0], sizeof(T)*weights.size(), cudaMemcpyHostToDevice, device.stream());
      persistent_initialized = true;
    }

    deltas_gpu_tensor = deltas_gpu.AccessTensor(context);
    weights_gpu_tensor = weights_gpu.AccessTensor(context);

    scaleinvariantgrad_grad_gpu( 
        device.stream(),
        output.data(),
        input.data(),
        gradient.data(),
        deltas_gpu_tensor->flat<int>().data(),
        weights_gpu_tensor->flat<T>().data(),
        deltas.size(),
        epsilon,
        input_shape.dim_size(input_rank-1),
        input_shape.dim_size(input_rank-2),
        w_size );
    
  }


  void scaleinvariantgrad_grad_gpu( 
      const cudaStream_t& stream,
      T* out, const T* in, const T* grad, 
      const int* deltas, const T* weights, int num_deltas, T eps,
      int x_size, int y_size, int z_size )
  {
    dim3 block(32,4,1);
    dim3 grid;
    grid.x = divup(x_size,block.x);
    grid.y = divup(y_size,block.y);
    grid.z = divup(z_size,block.z);

    computeBackward<T><<<grid,block,0,stream>>>( 
        out, in, grad, 
        x_size, y_size, z_size, eps,
        deltas, weights, num_deltas );
    CHECK_CUDA_ERROR
  }

private:
  std::vector<int> deltas;
  std::vector<T> weights;
  T epsilon;
  PersistentTensor deltas_gpu;
  PersistentTensor weights_gpu;
  bool persistent_initialized;

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ScaleInvariantGradientGrad")                                        \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ScaleInvariantGradientGradOp_GPU<type>);                                 
REG_KB(float)
REG_KB(double)
#undef REG_KB
