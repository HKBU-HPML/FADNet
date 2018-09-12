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
#include "rotation_format.h"
#include "Eigen/Geometry"


using namespace tensorflow;

REGISTER_OP("DepthToFlow")
  .Attr("T: {float, double}")
  .Attr("rotation_format: {'matrix', 'quaternion', 'angleaxis3'} = 'angleaxis3'")
  .Attr("inverse_depth: bool = false")
  .Attr("normalize_flow: bool = false")
  .Input("depth: T")
  .Input("intrinsics: T")
  .Input("rotation: T")
  .Input("translation: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle depth_shape, intrinsics_shape, rotation_shape, translation_shape, output_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &depth_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &intrinsics_shape));
      std::string rotation_format;
      c->GetAttr("rotation_format", &rotation_format);
      if( rotation_format == "matrix" )
      {
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 2, &rotation_shape));
      }
      else
      {
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &rotation_shape));
      }
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &translation_shape));

      if( c->RankKnown(intrinsics_shape) )
      {
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(intrinsics_shape,-1), 4, &d));
      }

      if( c->RankKnown(rotation_shape) )
      {
        DimensionHandle d;
        if( rotation_format == "matrix" )
        {
          TF_RETURN_IF_ERROR(c->WithValue(c->Dim(rotation_shape,-1), 3, &d));
          TF_RETURN_IF_ERROR(c->WithValue(c->Dim(rotation_shape,-2), 3, &d));
        }
        else if( rotation_format == "quaternion" )
        {
          TF_RETURN_IF_ERROR(c->WithValue(c->Dim(rotation_shape,-1), 4, &d));
        }
        else
        {
          TF_RETURN_IF_ERROR(c->WithValue(c->Dim(rotation_shape,-1), 3, &d));
        }
      }

      if( c->RankKnown(translation_shape) )
      {
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(translation_shape,-1), 3, &d));
      }

      if( c->RankKnown(depth_shape) )
      {
        ShapeHandle shape2d;
        c->Subshape(depth_shape, -2, &shape2d);

        ShapeHandle flow_shape;
        c->Concatenate(c->MakeShape({2}), shape2d, &flow_shape);


        int rank = c->Rank(depth_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-2; ++i )
        {
          c->Multiply(first_dim, c->Dim(depth_shape,i), &first_dim);
        }
        c->Concatenate(c->MakeShape({first_dim}), flow_shape, &output_shape);
        c->set_output(0, output_shape);
      }
      else
      {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    })
  .Doc(R"doc(
Computes the optical flow for an image pair based on the depth map and camera motion.

Takes the depth map of the first image and the relative camera motion of the
second image and computes the optical flow from the first to the second image.
The op assumes that the internal camera parameters are the same for both cameras.


depth: 
  depth map with absolute or inverse depth values
  The depth values describe the z distance to the optical center.

intrinsics:
  camera intrinsics in the format [fx, fy, cx, cy].
  fx,fy are the normalized focal lengths.
  cx,cy is the normalized position of the principal point.

rotation:
  The relative rotation R of the second camera.
  RX+t transforms a point X to the camera coordinate system of the second camera

translation:
  The relative translation vector t of the second camera.
  RX+t transforms a point X to the camera coordinate system of the second camera

rotation_format:
  The format for the rotation. 
  Allowed values are {'matrix', 'quaternion', 'angleaxis3'}.
  'matrix' is a 3x3 rotation matrix in column major order
  'quaternion' is a quaternion given as [w,x,y,z], w is the coefficient for the real part.
  'angleaxis3' is a 3d vector with the rotation axis. The angle is encoded as magnitude.

inverse_depth:
  If true then the input depth map must use inverse depth values.

normalize_flow:
  If true the returned optical flow will be normalized with respect to the 
  image dimensions.

output: 
  A tensor with the optical flow from the first to the second image.
  The format of the output tensor is NCHW with C=2; [batch, 2, height, width].
)doc");


namespace {

template <class T, class VEC2T, class VEC3T, class MAT3T>
inline void compute_flow( 
    Eigen::Matrix<T,2,1>& flow,        // the flow vector
    const Eigen::MatrixBase<VEC2T>& p1,        // pixel coordinates in the first image with pixel centers at x.5, y.5
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

} //namespace



template <class T>
class DepthToFlowOp : public OpKernel 
{
public:
  explicit DepthToFlowOp(OpKernelConstruction* construction)
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

    depthtoflow_cpu(
        output.data(),
        depth.data(),
        intrinsics.data(),
        rotation.data(),
        translation.data(),
        depth_shape.dim_size(depth_rank-1),
        depth_shape.dim_size(depth_rank-2),
        w_size );
    
  }


  void depthtoflow_cpu( 
      T* out, const T* depth, 
      const T* intrinsics,
      const T* rotation,
      const T* translation,
      int depth_x_size, int depth_y_size, int depth_z_size )
  {
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,3,3> Mat3;
    const int depth_xy_size = depth_x_size*depth_y_size;
    const T inv_depth_x_size = 1.0/depth_x_size;
    const T inv_depth_y_size = 1.0/depth_y_size;
    const int rotation_step = rotation_format_size(rotation_format);

    for( int z = 0; z < depth_z_size; ++z )
    {
      Vec2 f, c;
      f.x() = intrinsics[4*z+0]*depth_x_size;
      f.y() = intrinsics[4*z+1]*depth_y_size;
      c.x() = intrinsics[4*z+2]*depth_x_size;
      c.y() = intrinsics[4*z+3]*depth_y_size;
      Vec2 inv_f(1/f.x(), 1/f.y());

      Eigen::Map<const Vec3> t(translation+3*z);
      Mat3 R = convert_to_rotation_matrix(rotation+z*rotation_step, rotation_format);

      const T* depthmap = depth+z*depth_xy_size;
      T* flow = out+2*z*depth_xy_size;
#define DEPTH(x,y) depthmap[(y)*depth_x_size+(x)]
#define FLOW(c,x,y) flow[(c)*depth_xy_size+(y)*depth_x_size+(x)]
      for( int y = 0; y < depth_y_size; ++y )
      for( int x = 0; x < depth_x_size; ++x )
      {
        Vec2 flow_vec;

        T d = DEPTH(x,y);
        if( inverse_depth )
          d = 1/d;
        if( d > 0 && std::isfinite(d) )
        {
          Vec2 p1(x+T(0.5),y+T(0.5));
          compute_flow(flow_vec, p1, d, f, inv_f, c, R, t);
          if( normalize_flow  )
          {
            flow_vec.x() *= inv_depth_x_size;
            flow_vec.y() *= inv_depth_y_size;
          }
        }
        else
        {
          flow_vec.x() = std::numeric_limits<T>::quiet_NaN();
          flow_vec.y() = std::numeric_limits<T>::quiet_NaN();
        }

        FLOW(0,x,y) = flow_vec.x();
        FLOW(1,x,y) = flow_vec.y();
      }

    }
#undef DEPTH
#undef FLOW

  }


private:
  RotationFormat rotation_format;
  bool inverse_depth;
  bool normalize_flow;

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("DepthToFlow")                                                       \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    DepthToFlowOp<type>);                                                     
REG_KB(float)
REG_KB(double)
#undef REG_KB




