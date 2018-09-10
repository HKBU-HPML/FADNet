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
#include "rotation_format.h"
#include "Eigen/Core"
#include <cuda_runtime.h>

using namespace tensorflow;


namespace depthtoflow_internal
{

  template <class T, class VEC2T, class VEC3T, class MAT3T>
  __device__ inline void compute_flow( 
      Eigen::MatrixBase<VEC2T>& flow,        // the flow vector
      const Eigen::MatrixBase<VEC2T>& p1,    // pixel coordinates in the first image with pixel centers at x.5, y.5
      const T depth,                         // depth of the point in the first image
      const Eigen::MatrixBase<VEC2T>& f,     // focal lengths
      const Eigen::MatrixBase<VEC2T>& inv_f, // reciprocal of focal lengths (1/f.x, 1/f.y)
      const Eigen::MatrixBase<VEC2T>& c,     // principal point coordinates, not pixel coordinates! pixel centers are shifted by 0.5
      const Eigen::MatrixBase<MAT3T>& R,     // rotation
      const Eigen::MatrixBase<VEC3T>& t      // translation
      ) 
  {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VEC2T, 2) 
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VEC3T, 3) 
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(MAT3T, 3, 3) 
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<T,3,1> Vec3;
    // compute the 3d point in the coordinate frame of the first camera
    Vec2 tmp2 = (p1-c).cwiseProduct(inv_f);

    // transform the point to the coordinate frame of the second camera
    Vec3 p2 = R*(depth*tmp2.homogeneous()) + t;
    
    // project point to the image plane
    p2.x() = f.x()*(p2.x()/p2.z()) + c.x();
    p2.y() = f.y()*(p2.y()/p2.z()) + c.y();
    flow = p2.template topRows<2>() - p1;
  }

  template <class T, bool NORMALIZE_FLOW, bool INVERSE_DEPTH>
  __global__ void depthtoflow_kernel(
      T* out, const T* depth,
      const T* intrinsics,
      const T* rotation,
      const T* translation,
      int depth_x_size, int depth_y_size, int depth_z_size, int depth_xy_size,
      T inv_depth_x_size, T inv_depth_y_size )
  {
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,3,3> Mat3;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= depth_x_size || y >= depth_y_size || z >= depth_z_size )
      return;

    Vec2 f, c;
    if( NORMALIZE_FLOW )
    {
      f.x() = intrinsics[4*z+0];
      f.y() = intrinsics[4*z+1];
      c.x() = intrinsics[4*z+2];
      c.y() = intrinsics[4*z+3];
    }
    else
    {
      f.x() = intrinsics[4*z+0]*depth_x_size;
      f.y() = intrinsics[4*z+1]*depth_y_size;
      c.x() = intrinsics[4*z+2]*depth_x_size;
      c.y() = intrinsics[4*z+3]*depth_y_size;
    }
    Vec2 inv_f(1/f.x(), 1/f.y());

    Eigen::Map<const Vec3> t(translation+3*z);
    Eigen::Map<const Mat3> R(rotation+9*z);

    const T* depthmap = depth+z*depth_xy_size;
    T* flow = out+2*z*depth_xy_size;
#define DEPTH(x,y) depthmap[(y)*depth_x_size+(x)]
#define FLOW(c,x,y) flow[(c)*depth_xy_size+(y)*depth_x_size+(x)]
    {
      Vec2 flow_vec(NAN,NAN);

      T d = DEPTH(x,y);
      if( INVERSE_DEPTH )
        d = 1/d;
      if( d > 0 && isfinite(d) )
      {
        Vec2 p1(x+T(0.5),y+T(0.5));
        if( NORMALIZE_FLOW )
        {
          p1.x() *= inv_depth_x_size;
          p1.y() *= inv_depth_y_size;
        }
        compute_flow(flow_vec, p1, d, f, inv_f, c, R, t);
      }

      FLOW(0,x,y) = flow_vec.x();
      FLOW(1,x,y) = flow_vec.y();
    }
#undef DEPTH
#undef FLOW
  }


}
using namespace depthtoflow_internal;



template <class T>
void depthtoflow_gpu( 
      const cudaStream_t& stream,
      T* out, 
      const T* depth, 
      const T* intrinsics,
      const T* rotation,
      const T* translation,
      int depth_x_size, int depth_y_size, int depth_z_size,
      bool normalize_flow,
      bool inverse_depth )
{
  dim3 block(32,4,1);
  dim3 grid;
  grid.x = divup(depth_x_size,block.x);
  grid.y = divup(depth_y_size,block.y);
  grid.z = divup(depth_z_size,block.z);

  if( normalize_flow )
  {
    if( inverse_depth )
    {
      depthtoflow_kernel<T,true,true><<<grid,block,0,stream>>>(
          out, depth,
          intrinsics,
          rotation,
          translation,
          depth_x_size, depth_y_size, depth_z_size, depth_x_size*depth_y_size,
          1.0/depth_x_size, 1.0/depth_y_size );
      CHECK_CUDA_ERROR
    }
    else
    {
      depthtoflow_kernel<T,true,false><<<grid,block,0,stream>>>(
          out, depth,
          intrinsics,
          rotation,
          translation,
          depth_x_size, depth_y_size, depth_z_size, depth_x_size*depth_y_size,
          1.0/depth_x_size, 1.0/depth_y_size );
      CHECK_CUDA_ERROR
    }
  }
  else
  {
    if( inverse_depth )
    {
      depthtoflow_kernel<T,false,true><<<grid,block,0,stream>>>(
          out, depth,
          intrinsics,
          rotation,
          translation,
          depth_x_size, depth_y_size, depth_z_size, depth_x_size*depth_y_size,
          1.0/depth_x_size, 1.0/depth_y_size );
      CHECK_CUDA_ERROR
    }
    else
    {
      depthtoflow_kernel<T,false,false><<<grid,block,0,stream>>>(
          out, depth,
          intrinsics,
          rotation,
          translation,
          depth_x_size, depth_y_size, depth_z_size, depth_x_size*depth_y_size,
          1.0/depth_x_size, 1.0/depth_y_size );
      CHECK_CUDA_ERROR
    }
  }
}
template void depthtoflow_gpu<float>(const cudaStream_t&, float*, const float*, const float*, const float*, const float*, int, int, int, bool, bool);
template void depthtoflow_gpu<double>(const cudaStream_t&, double*, const double*, const double*, const double*, const double*, int, int, int, bool, bool);


template <class T>
class DepthToFlowOp_GPU : public OpKernel 
{
public:
  explicit DepthToFlowOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
    std::string R_format;
    OP_REQUIRES_OK(construction, construction->GetAttr("rotation_format", &R_format));
    if( R_format == "matrix" )
      rotation_format = MATRIX;
    else if( R_format == "quaternion" )
      rotation_format = QUATERNION;
    else
      rotation_format = ANGLEAXIS3;

    OP_REQUIRES_OK(construction, construction->GetAttr("inverse_depth", &inverse_depth));
    OP_REQUIRES_OK(construction, construction->GetAttr("normalize_flow", &normalize_flow));
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& depth_tensor = context->input(0);
    auto depth = depth_tensor.flat<T>();
    const TensorShape depth_shape(depth_tensor.shape());
    const int depth_rank = depth_shape.dims();

    const Tensor& intrinsics_tensor = context->input(1);
    auto intrinsics = intrinsics_tensor.flat<T>();
    const Tensor& rotation_tensor = context->input(2);
    auto rotation = rotation_tensor.flat<T>();
    const Tensor& translation_tensor = context->input(3);
    auto translation = translation_tensor.flat<T>();

    TensorShape output_shape;
    int64_t w_size = 1;
    for( int i = 0; i < depth_rank-2; ++i )
      w_size *= depth_shape.dim_size(i);
    output_shape.AddDim(w_size);
    output_shape.AddDim(2);
    output_shape.AddDim(depth_shape.dim_size(depth_rank-2));
    output_shape.AddDim(depth_shape.dim_size(depth_rank-1));
    Tensor* output_tensor = 0; 
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    auto device = context->eigen_gpu_device();
    if( rotation_format == MATRIX )
    {
      depthtoflow_gpu( 
            device.stream(),
            output.data(),
            depth.data(),
            intrinsics.data(),
            rotation_tensor.flat<T>().data(),
            translation.data(),
            depth_shape.dim_size(depth_rank-1),
            depth_shape.dim_size(depth_rank-2),
            w_size,
            normalize_flow,
            inverse_depth );
    }
    else if( rotation_format == ANGLEAXIS3 )
    {
      TensorShape rotmatrix_shape(rotation_tensor.shape());
      rotmatrix_shape.set_dim(rotmatrix_shape.dims()-1, 9);

      Tensor rotmatrix_tensor_gpu;
      OP_REQUIRES_OK(context, 
          context->allocate_temp( DataTypeToEnum<T>::v(), 
            rotmatrix_shape, 
            &rotmatrix_tensor_gpu));
      
      T *out_gpu = rotmatrix_tensor_gpu.flat<T>().data();
      const T *in_gpu = rotation_tensor.flat<T>().data();
      angleaxis_to_rotmatrix_gpu(device.stream(), out_gpu, in_gpu, w_size);

      depthtoflow_gpu( 
            device.stream(),
            output.data(),
            depth.data(),
            intrinsics.data(),
            rotmatrix_tensor_gpu.flat<T>().data(),
            translation.data(),
            depth_shape.dim_size(depth_rank-1),
            depth_shape.dim_size(depth_rank-2),
            w_size,
            normalize_flow,
            inverse_depth );
    }
    else
    {
      // convert to rotation matrix on the cpu
      AllocatorAttributes attr;
      attr.set_on_host(true);
      attr.set_gpu_compatible(true);
      
      Tensor rotation_tensor_cpu;
      OP_REQUIRES_OK(context, 
          context->allocate_temp( DataTypeToEnum<T>::v(), 
            rotation_tensor.shape(), 
            &rotation_tensor_cpu,
            attr));

      TensorShape rotmatrix_shape(rotation_tensor.shape());
      rotmatrix_shape.set_dim(rotmatrix_shape.dims()-1, 9);
      Tensor rotmatrix_tensor_cpu;
      OP_REQUIRES_OK(context, 
          context->allocate_temp( DataTypeToEnum<T>::v(), 
            rotmatrix_shape, 
            &rotmatrix_tensor_cpu,
            attr));

      Tensor rotmatrix_tensor_gpu;
      OP_REQUIRES_OK(context, 
          context->allocate_temp( DataTypeToEnum<T>::v(), 
            rotmatrix_shape, 
            &rotmatrix_tensor_gpu));

      {
        typedef Eigen::Matrix<T,3,3> Mat3;
        const int step = rotation_format_size(rotation_format);
        const T *in_gpu = rotation_tensor.flat<T>().data();
        T *in_cpu = rotation_tensor_cpu.flat<T>().data();
        T *out_cpu = rotmatrix_tensor_cpu.flat<T>().data();
        T *out_gpu = rotmatrix_tensor_gpu.flat<T>().data();
        //device.memcpyDeviceToHost(in_cpu, in_gpu, sizeof(T)*w_size*step); // Is this async?
        cudaMemcpyAsync(in_cpu, in_gpu, sizeof(T)*w_size*step, cudaMemcpyDeviceToHost, device.stream() );
        cudaStreamSynchronize(device.stream());
        for( int i = 0; i < w_size; ++i )
        {
          Mat3 R = convert_to_rotation_matrix(in_cpu+step*i, rotation_format);
          Eigen::Map<Mat3> tmp(out_cpu+9*i);
          tmp = R;
        }
        //device.memcpyHostToDevice(out_gpu, out_cpu, sizeof(T)*w_size*9);
        cudaMemcpyAsync(out_gpu, out_cpu, sizeof(T)*w_size*9, cudaMemcpyHostToDevice, device.stream());

      }
      depthtoflow_gpu( 
            device.stream(),
            output.data(),
            depth.data(),
            intrinsics.data(),
            rotmatrix_tensor_gpu.flat<T>().data(),
            translation.data(),
            depth_shape.dim_size(depth_rank-1),
            depth_shape.dim_size(depth_rank-2),
            w_size,
            normalize_flow,
            inverse_depth );
    }

    
  }


private:
  RotationFormat rotation_format;
  bool inverse_depth;
  bool normalize_flow;

};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("DepthToFlow")                                                       \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    DepthToFlowOp_GPU<type>);                                                  
REG_KB(float)
REG_KB(double)
#undef REG_KB

