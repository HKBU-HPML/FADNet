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

namespace median3x3downsample_internal
{
  template <class T>
  __global__ void median3x3downsample_kernel(
      T* out, const T* in,
      int z_size, 
      int out_x_size, int out_y_size, int out_xy_size,
      int in_x_size, int in_y_size, int in_xy_size )
  {
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= out_x_size || y >= out_y_size || z >= z_size )
      return;

    T value[9];
    int value_idx = 0;
    for( int dy = -1; dy <= 1; ++dy )
    for( int dx = -1; dx <= 1; ++dx )
    {
      int x_ = min(in_x_size-1,max(0,2*x+dx));
      int y_ = min(in_y_size-1,max(0,2*y+dy));
      value[value_idx++] = in[z*in_xy_size+y_*in_x_size+x_];
    }
    {
      for(int j = 1; j < 9; ++j)
      {
        if( value[0] > value[j] )
        {
          T tmp = value[0];
          value[0] = value[j];
          value[j] = tmp;
        }
      }
      for(int j = 2; j < 9; ++j)
      {
        if( value[1] > value[j] )
        {
          T tmp = value[1];
          value[1] = value[j];
          value[j] = tmp;
        }
      }
      for(int j = 3; j < 9; ++j)
      {
        if( value[2] > value[j] )
        {
          T tmp = value[2];
          value[2] = value[j];
          value[j] = tmp;
        }
      }
      for(int j = 4; j < 9; ++j)
      {
        if( value[3] > value[j] )
        {
          T tmp = value[3];
          value[3] = value[j];
          value[j] = tmp;
        }
      }
      for(int j = 5; j < 9; ++j)
      {
        if( value[4] > value[j] )
        {
          T tmp = value[4];
          value[4] = value[j];
          value[j] = tmp;
        }
      }
    }
    int out_idx = z*out_xy_size + y*out_x_size + x;
    out[out_idx] = value[4];
  }
}
using namespace median3x3downsample_internal;


template <class T>
class Median3x3DownsampleOp_GPU : public OpKernel 
{
public:
  explicit Median3x3DownsampleOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction)
  { }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());
    const int rank = input_shape.dims();
    TensorShape output_shape(input_tensor.shape());
    {
      int idx = rank-1;
      output_shape.set_dim(idx,divup(output_shape.dim_size(idx),2));
      idx = rank-2;
      output_shape.set_dim(idx,divup(output_shape.dim_size(idx),2));
    }
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    int64_t z_size = 1;
    for( int i = 0; i < rank-2; ++i )
      z_size *= output_shape.dim_size(i);
    
    auto device = context->eigen_gpu_device();
    median3x3downsample_gpu(
        device.stream(),
        output.data(), input.data(), 
        z_size, 
        input_shape.dim_size(rank-2),
        input_shape.dim_size(rank-1)
        );
  }

  void median3x3downsample_gpu( 
      const cudaStream_t& stream,
      T* out, const T* in, 
      int z_size, 
      int in_y_size, int in_x_size )
  {
    int out_x_size = divup(in_x_size,2);
    int out_y_size = divup(in_y_size,2);
    int out_xy_size = out_x_size*out_y_size;
    int in_xy_size = in_x_size*in_y_size;
    dim3 block(32,4,1);
    dim3 grid;
    grid.x = divup(out_x_size,block.x);
    grid.y = divup(out_y_size,block.y);
    grid.z = divup(z_size,block.z);

    median3x3downsample_kernel<T><<<grid,block,0,stream>>>(
        out, in, 
        z_size,
        out_x_size, out_y_size, out_xy_size,
        in_x_size, in_y_size, in_xy_size
        );
  }

private:

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("Median3x3Downsample")                                               \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    Median3x3DownsampleOp_GPU<type>);                                         
REG_KB(float)
REG_KB(double)
#undef REG_KB

