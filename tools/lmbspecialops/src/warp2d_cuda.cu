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
#include "Eigen/Core"

using namespace tensorflow;

namespace warp2d_kernel_internal
{
#define CLAMP 1
#define VALUE 2
  template <class T, bool NORMALIZED, int BORDER_MODE>
  __global__ void warp2d_kernel(
      T* out, const T* in, const T* displacements, const T border_value,
      int x_size, int y_size, int z_size, int w_size)
  {
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= x_size || y >= y_size || w >= w_size )
      return;

    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<int,2,1> Vec2i;
    typedef Eigen::Matrix<T,4,1> Vec4;
    const int xy_size = x_size*y_size;
    const int xyz_size = xy_size*z_size;
#define IN(w,z,y,x) in[(w)*xyz_size+(z)*xy_size+(y)*x_size+(x)]
#define OUT(w,z,y,x) out[(w)*xyz_size+(z)*xy_size+(y)*x_size+(x)]
#define VECTOR(w,z,y,x) displacements[(w)*2*xy_size+(z)*xy_size+(y)*x_size+(x)]
    Vec2 p1(x,y);
    Vec2 v(VECTOR(w,0,y,x), VECTOR(w,1,y,x));
    if( NORMALIZED )
    {
      v.x() *= x_size;
      v.y() *= y_size;
    }
    Vec2 p2 = p1+v;
    Vec2i p2i = p2.template cast<int>();
    
    T a = p2.x()-p2i.x();
    T b = p2.y()-p2i.y();
    Vec4 weights( (1-a)*(1-b), a*(1-b), (1-a)*b, a*b );
    Vec4 values;

    if( BORDER_MODE == CLAMP )
    {
      int x0, y0, x1, y1, x2, y2, x3, y3;
      x0 = min(x_size-1,max(0,p2i.x()));
      y0 = min(y_size-1,max(0,p2i.y()));
      x1 = min(x_size-1,max(0,p2i.x()+1));
      y1 = min(y_size-1,max(0,p2i.y()));
      x2 = min(x_size-1,max(0,p2i.x()));
      y2 = min(y_size-1,max(0,p2i.y()+1));
      x3 = min(x_size-1,max(0,p2i.x()+1));
      y3 = min(y_size-1,max(0,p2i.y()+1));
      for( int z = 0; z < z_size; ++z )
      {
        values(0) = IN(w,z,y0,x0);
        values(1) = IN(w,z,y1,x1);
        values(2) = IN(w,z,y2,x2);
        values(3) = IN(w,z,y3,x3);
        OUT(w,z,y,x) = values.dot(weights);
      }
    }
    else
    {
      int x0, y0, x1, y1, x2, y2, x3, y3;
      x0 = p2i.x();
      y0 = p2i.y();
      x1 = p2i.x()+1;
      y1 = p2i.y();
      x2 = p2i.x();
      y2 = p2i.y()+1;
      x3 = p2i.x()+1;
      y3 = p2i.y()+1;
      for( int z = 0; z < z_size; ++z )
      {
        if( x0 >= 0 && x3 > 0 && x3 < x_size && y0 >= 0 && y3 > 0 && y3 < y_size )
        {
          values(0) = IN(w,z,y0,x0);
          values(1) = IN(w,z,y1,x1);
          values(2) = IN(w,z,y2,x2);
          values(3) = IN(w,z,y3,x3);
          OUT(w,z,y,x) = values.dot(weights);
        }
        else
        {
          OUT(w,z,y,x) = border_value;
        }
      }
    }
#undef IN
#undef OUT
#undef VECTOR
  }

} 
using namespace warp2d_kernel_internal;




template <class T>
class Warp2dOp_GPU : public OpKernel 
{
public:
  explicit Warp2dOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
    OP_REQUIRES_OK(construction, construction->GetAttr("normalized", &normalized));
    float value_tmp;
    OP_REQUIRES_OK(construction, construction->GetAttr("border_value", &value_tmp));
    border_value = value_tmp;
    std::string border_mode_str;
    OP_REQUIRES_OK(construction, construction->GetAttr("border_mode", &border_mode_str));
    if( border_mode_str == "clamp" )
      border_mode = CLAMP;
    else 
      border_mode = VALUE;
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());

    const Tensor& displacements_tensor = context->input(1);
    auto displacements = displacements_tensor.flat<T>();
    const TensorShape displacements_shape(displacements_tensor.shape());

    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    const int input_rank = input_shape.dims();


    int x_size = input_shape.dim_size(input_rank-1);
    int y_size = input_shape.dim_size(input_rank-2);
    int z_size = 1;
    int w_size = 1;
    if( input_rank >= 3 )
      z_size = input_shape.dim_size(input_rank-3);
    for( int i = 0; i < input_rank-3; ++i )
      w_size *= input_shape.dim_size(i);


    auto device = context->eigen_gpu_device();
    warp2d_gpu( 
        device.stream(),
        output.data(),
        input.data(),
        displacements.data(),
        border_value,
        border_mode,
        normalized,
        x_size, y_size, z_size, w_size );
    
  }

  void warp2d_gpu( 
      const cudaStream_t& stream,
      T* out, const T* in, const T* displacements,
      const T border_value, const int border_mode, bool normalized,
      int x_size, int y_size, int z_size, int w_size)
  {

    dim3 block(32,4,1);
    dim3 grid;
    grid.x = divup(x_size,block.x);
    grid.y = divup(y_size,block.y);
    grid.z = divup(w_size,block.z);

    if( normalized )
    {
      if( border_mode == CLAMP )
        warp2d_kernel<T,true,CLAMP><<<grid,block,0,stream>>>(
            out, in, displacements, border_value,
            x_size, y_size, z_size, w_size);
      else
        warp2d_kernel<T,true,VALUE><<<grid,block,0,stream>>>(
            out, in, displacements, border_value,
            x_size, y_size, z_size, w_size);
    }
    else
    {
      if( border_mode == CLAMP )
        warp2d_kernel<T,false,CLAMP><<<grid,block,0,stream>>>(
            out, in, displacements, border_value,
            x_size, y_size, z_size, w_size);
      else
        warp2d_kernel<T,false,VALUE><<<grid,block,0,stream>>>(
            out, in, displacements, border_value,
            x_size, y_size, z_size, w_size);
    }

    CHECK_CUDA_ERROR;
  }



private:
  int border_mode;
  T border_value;
  bool normalized;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("Warp2d")                                                            \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    Warp2dOp_GPU<type>);                                                    
REG_KB(float)
REG_KB(double)
#undef REG_KB

