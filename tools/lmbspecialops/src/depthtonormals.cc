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
#include "helper.h"
#include "Eigen/Dense"


using namespace tensorflow;

REGISTER_OP("DepthToNormals")
  .Attr("T: {float, double}")
  .Attr("inverse_depth: bool = false")
  .Input("depth: T")
  .Input("intrinsics: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle depth_shape, intrinsics_shape, output_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &depth_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &intrinsics_shape));

      if( c->RankKnown(intrinsics_shape) )
      {
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(intrinsics_shape,-1), 4, &d));
      }

      if( c->RankKnown(depth_shape) )
      {
        ShapeHandle shape2d;
        c->Subshape(depth_shape, -2, &shape2d);

        ShapeHandle normal_shape;
        c->Concatenate(c->MakeShape({3}), shape2d, &normal_shape);

        int rank = c->Rank(depth_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-2; ++i )
        {
          c->Multiply(first_dim, c->Dim(depth_shape,i), &first_dim);
        }
        c->Concatenate(c->MakeShape({first_dim}), normal_shape, &output_shape);
        c->set_output(0, output_shape);
      }
      else
      {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    })
  .Doc(R"doc(
Computes the normal map from a depth map.

depth: 
  depth map with absolute or inverse depth values
  The depth values describe the z distance to the optical center.

intrinsics:
  camera intrinsics in the format [fx, fy, cx, cy].
  fx,fy are the normalized focal lengths.
  cx,cy is the normalized position of the principal point.

inverse_depth:
  If true then the input depth map must use inverse depth values.

output: 
  Normal map in the coordinate system of the camera.
  The format of the output tensor is NCHW with C=3; [batch, 3, height, width].
)doc");


namespace {

template <class T>
void compute3dPoint( Eigen::Matrix<T,3,1>& p_3d, const int x, const int y, const T depth, const Eigen::Matrix<T,3,3>& inv_K)
{
  p_3d << ((x+T(0.5))*inv_K(0,0) + inv_K(0,2))*depth, 
          ((y+T(0.5))*inv_K(1,1) + inv_K(1,2))*depth, 
          depth;
}

} //namespace


template <class T>
class DepthToNormalsOp : public OpKernel 
{
public:
  explicit DepthToNormalsOp(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
    OP_REQUIRES_OK(construction, construction->GetAttr("inverse_depth", &inverse_depth));
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& depth_tensor = context->input(0);
    auto depth = depth_tensor.flat<T>();
    const TensorShape depth_shape(depth_tensor.shape());
    const int depth_rank = depth_shape.dims();

    const Tensor& intrinsics_tensor = context->input(1);
    auto intrinsics = intrinsics_tensor.flat<T>();

    TensorShape output_shape;
    int64_t w_size = 1;
    for( int i = 0; i < depth_rank-2; ++i )
      w_size *= depth_shape.dim_size(i);
    output_shape.AddDim(w_size);
    output_shape.AddDim(3);
    output_shape.AddDim(depth_shape.dim_size(depth_rank-2));
    output_shape.AddDim(depth_shape.dim_size(depth_rank-1));
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    depthtonormals_cpu(
        output.data(),
        depth.data(),
        intrinsics.data(),
        depth_shape.dim_size(depth_rank-1),
        depth_shape.dim_size(depth_rank-2),
        w_size );
    
  }


  void depthtonormals_cpu( 
      T* out, const T* depth, 
      const T* intrinsics,
      int x_size, int y_size, int z_size )
  {
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,3,3> Mat3;
    const int xy_size = x_size*y_size;

    for( int z = 0; z < z_size; ++z )
    {
      Mat3 K(Mat3::Identity());
      K(0,0) = intrinsics[4*z+0]*x_size;
      K(1,1) = intrinsics[4*z+1]*y_size;
      K(0,2) = intrinsics[4*z+2]*x_size;
      K(1,2) = intrinsics[4*z+3]*y_size;
      Mat3 inv_K = K.inverse();
      


      const T* depthmap = depth+z*xy_size;
      T* normal = out+3*z*xy_size;
#define DEPTH(y,x) depthmap[(y)*x_size+(x)]
#define NORMAL(c,y,x) normal[(c)*xy_size+(y)*x_size+(x)]
      for( int y = 0; y < y_size; ++y )
      for( int x = 0; x < x_size; ++x )
      {
        if( x == 0 || y == 0 || x == x_size-1 || y == y_size-1)
        {
          NORMAL(0,y,x) = std::numeric_limits<T>::quiet_NaN();
          NORMAL(1,y,x) = std::numeric_limits<T>::quiet_NaN();
          NORMAL(2,y,x) = std::numeric_limits<T>::quiet_NaN();
        }
        else
        {
          T d = DEPTH(y,x);
          T d_y0 = DEPTH(y-1,x);
          T d_x0 = DEPTH(y,x-1);
          T d_y1 = DEPTH(y+1,x);
          T d_x1 = DEPTH(y,x+1);
          if( inverse_depth )
          {
            d = 1/d;
            d_y0 = 1/d_y0;
            d_x0 = 1/d_x0;
            d_y1 = 1/d_y1;
            d_x1 = 1/d_x1;
          }
          
          if( d <= 0 || !std::isfinite(d) || 
              d_y0 <= 0 || !std::isfinite(d_y0) || 
              d_x0 <= 0 || !std::isfinite(d_x0) || 
              d_y1 <= 0 || !std::isfinite(d_y1) || 
              d_x1 <= 0 || !std::isfinite(d_x1))
          {
            NORMAL(0,y,x) = std::numeric_limits<T>::quiet_NaN();
            NORMAL(1,y,x) = std::numeric_limits<T>::quiet_NaN();
            NORMAL(2,y,x) = std::numeric_limits<T>::quiet_NaN();
          }
          else
          {
            Vec3 p, p_y0, p_x0, p_y1, p_x1;
            compute3dPoint(p, x, y, d, inv_K);
            compute3dPoint(p_y0, x, y-1, d_y0, inv_K);
            compute3dPoint(p_x0, x-1, y, d_x0, inv_K);
            compute3dPoint(p_y1, x, y+1, d_y1, inv_K);
            compute3dPoint(p_x1, x+1, y, d_x1, inv_K);

            Vec3 normals_vec1 = (p - p_x1).cross(p_y1 - p);
            Vec3 normals_vec0 = (p - p_x0).cross(p_y0 - p);
            normals_vec1.normalize();
            normals_vec0.normalize();
            
            Vec3 normals_vec = (normals_vec1 + normals_vec0);
            normals_vec.normalize();
            
            NORMAL(0,y,x) = normals_vec.x();
            NORMAL(1,y,x) = normals_vec.y();
            NORMAL(2,y,x) = normals_vec.z();
          }
        }

      }

    }
#undef DEPTH
#undef NORMAL

  }


private:
  bool inverse_depth;

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("DepthToNormals")                                                    \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    DepthToNormalsOp<type>);                                             
REG_KB(float)
REG_KB(double)
#undef REG_KB


