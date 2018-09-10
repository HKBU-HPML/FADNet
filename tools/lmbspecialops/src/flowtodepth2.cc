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

REGISTER_OP("FlowToDepth2")
  .Attr("T: {float, double}")
  .Attr("rotation_format: {'matrix', 'quaternion', 'angleaxis3'} = 'angleaxis3'")
  .Attr("inverse_depth: bool = false")
  .Attr("normalized_flow: bool = false")
  .Input("flow: T")
  .Input("intrinsics: T")
  .Input("rotation: T")
  .Input("translation: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle flow_shape, intrinsics_shape, rotation_shape, translation_shape, output_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 3, &flow_shape));
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

      if( c->RankKnown(flow_shape) )
      {
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(flow_shape,-3), 2, &d));

        ShapeHandle shape2d;
        c->Subshape(flow_shape, -2, &shape2d);

        int rank = c->Rank(flow_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-3; ++i )
        {
          c->Multiply(first_dim, c->Dim(flow_shape,i), &first_dim);
        }
        c->Concatenate(c->MakeShape({first_dim, c->MakeDim(1)}), shape2d, &output_shape);
        c->set_output(0, output_shape);


        // check if the batch dimension matches with the other parameters
        if( c->RankKnown(intrinsics_shape) )
        {
          DimensionHandle intrinsics_first_dim = c->MakeDim(1);
          int rank = c->Rank(intrinsics_shape);
          for( int i = 0; i < rank-1; ++i )
          {
            c->Multiply(intrinsics_first_dim, c->Dim(intrinsics_shape,i), &intrinsics_first_dim);
          }
          TF_RETURN_IF_ERROR(c->Merge(first_dim, intrinsics_first_dim, &first_dim));
        }
        if( c->RankKnown(translation_shape) )
        {
          DimensionHandle translation_first_dim = c->MakeDim(1);
          int rank = c->Rank(translation_shape);
          for( int i = 0; i < rank-1; ++i )
          {
            c->Multiply(translation_first_dim, c->Dim(translation_shape,i), &translation_first_dim);
          }
          TF_RETURN_IF_ERROR(c->Merge(first_dim, translation_first_dim, &first_dim));
        }
        if( c->RankKnown(rotation_shape) )
        {
          DimensionHandle rotation_first_dim = c->MakeDim(1);
          int rank = c->Rank(rotation_shape);
          int min_rank = 1;
          if( rotation_format == "matrix" )
            min_rank = 2;
          for( int i = 0; i < rank-min_rank; ++i )
          {
            c->Multiply(rotation_first_dim, c->Dim(rotation_shape,i), &rotation_first_dim);
          }
          TF_RETURN_IF_ERROR(c->Merge(first_dim, rotation_first_dim, &first_dim));
        }

      }
      else
      {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    })
  .Doc(R"doc(
Computes the depth from optical flow and the camera motion.

Takes the optical flow and the relative camera motion from the second camera to
compute a depth map.
The layer assumes that the internal camera parameters are the same for both
images.


flow:
  optical flow normalized or in pixel units. The tensor format must be NCHW.

intrinsics:
  camera intrinsics in the format [fx, fy, cx, cy].
  fx,fy are the normalized focal lengths.
  cx,cy is the normalized position of the principal point.
  The format of the tensor is NC with C=4 and N matching the batch size of the 
  flow tensor.
  

rotation:
  The relative rotation R of the second camera.
  RX+t transforms a point X to the camera coordinate system of the second camera
  The format of the tensor is either NC with C=3 or C=4 or NIJ with I=3 and J=3.
  N matches the batch size of the flow tensor.
  
  
translation:
  The relative translation vector t of the second camera.
  RX+t transforms a point X to the camera coordinate system of the second camera
  The format of the tensor is NC with C=3 and N matching the batch size of the 
  flow tensor.

rotation_format:
  The format for the rotation. 
  Allowed values are {'matrix', 'quaternion', 'angleaxis3'}.
  'angleaxis3' is a 3d vector with the rotation axis. 
  The angle is encoded as magnitude.

inverse_depth:
  If true then the output depth map uses inverse depth values.

normalized_flow:
  If true then the input flow is expected to be normalized with respect to the 
  image dimensions.

output:
  A tensor with the depth for the first image.
  The format of the output tensor is NCHW with C=1; [batch, 1, height, width].

)doc");


namespace
{

template <class T>
using std_vector_Vector2 = std::vector<Eigen::Matrix<T,2,1>,Eigen::aligned_allocator<Eigen::Matrix<T,2,1> > >;

template <class T>
using std_vector_Matrix3x4 = std::vector<Eigen::Matrix<T,3,4>,Eigen::aligned_allocator<Eigen::Matrix<T,3,4> > >;


template <class T>
Eigen::Matrix<T,3,3> computeFundamentalFromCameras(
                                      const Eigen::Matrix<T,3,4>& P1, 
                                      const Eigen::Matrix<T,3,4>& P2)
{
  Eigen::Matrix<T,3,3> F;
  Eigen::Matrix<T,2,4> X1, X2, X3;
  X1 = P1.bottomRows(2);
  X2.topRows(1) = P1.bottomRows(1);
  X2.bottomRows(1) = P1.topRows(1);
  X3 = P1.topRows(2);

  Eigen::Matrix<T,2,4> Y1, Y2, Y3;
  Y1 = P2.bottomRows(2);
  Y2.topRows(1) = P2.bottomRows(1);
  Y2.bottomRows(1) = P2.topRows(1);
  Y3 = P2.topRows(2);

  Eigen::Matrix<T,4,4> tmp;
  tmp << X1, Y1;
  F(0,0) = tmp.determinant();
  tmp << X2, Y1;
  F(0,1) = tmp.determinant();
  tmp << X3, Y1;
  F(0,2) = tmp.determinant();
  
  tmp << X1, Y2;
  F(1,0) = tmp.determinant();
  tmp << X2, Y2;
  F(1,1) = tmp.determinant();
  tmp << X3, Y2;
  F(1,2) = tmp.determinant();
  
  tmp << X1, Y3;
  F(2,0) = tmp.determinant();
  tmp << X2, Y3;
  F(2,1) = tmp.determinant();
  tmp << X3, Y3;
  F(2,2) = tmp.determinant();
  
  return F;
}


template <class T>
Eigen::Matrix<T,3,1> triangulateLinear(
    const std_vector_Matrix3x4<T>& Pvec,
    const std_vector_Vector2<T>& xvec )
{
  int n = Pvec.size();

  Eigen::Matrix<T,Eigen::Dynamic,3> A(2*n,3);
  Eigen::Matrix<T,Eigen::Dynamic,1> b(2*n);

  for( int i = 0; i < n; ++i )
  {
    T x = xvec[i].x();
    T y = xvec[i].y();
    T z = 1;

    const Eigen::Matrix<T,3,4>& P = Pvec[i];
    A.row(2*i+0) = y*P.block(2,0,1,3) - z*P.block(1,0,1,3); 
    A.row(2*i+1) = z*P.block(0,0,1,3) - x*P.block(2,0,1,3);

    b(2*i+0) = z*P(1,3) - y*P(2,3);
    b(2*i+1) = x*P(2,3) - z*P(0,3);
  }

  Eigen::JacobiSVD<Eigen::Matrix<T,Eigen::Dynamic,3> > svd(A,Eigen::ComputeFullU|Eigen::ComputeFullV);

  Eigen::Matrix<T,3,1> X; 
  X = svd.solve( b );

  return X;
}


}


template <class T>
class FlowToDepth2Op : public OpKernel 
{
public:
  explicit FlowToDepth2Op(OpKernelConstruction* construction)
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
    OP_REQUIRES_OK(construction, construction->GetAttr("normalized_flow", &normalized_flow));
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& flow_tensor = context->input(0);
    auto flow = flow_tensor.flat<T>();
    const TensorShape flow_shape(flow_tensor.shape());
    const int flow_rank = flow_shape.dims();

    const Tensor& intrinsics_tensor = context->input(1);
    auto intrinsics = intrinsics_tensor.flat<T>();
    const Tensor& rotation_tensor = context->input(2);
    auto rotation = rotation_tensor.flat<T>();
    const Tensor& translation_tensor = context->input(3);
    auto translation = translation_tensor.flat<T>();

    TensorShape output_shape;
    int64_t w_size = 1;
    for( int i = 0; i < flow_rank-3; ++i )
      w_size *= flow_shape.dim_size(i);
    output_shape.AddDim(w_size);
    output_shape.AddDim(1);
    output_shape.AddDim(flow_shape.dim_size(flow_rank-2));
    output_shape.AddDim(flow_shape.dim_size(flow_rank-1));
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    if( inverse_depth )
    {
      if( normalized_flow )
        flowtodepth_cpu<true,true>(
            output.data(),
            flow.data(),
            intrinsics.data(),
            rotation.data(),
            translation.data(),
            flow_shape.dim_size(flow_rank-1),
            flow_shape.dim_size(flow_rank-2),
            w_size );
      else
        flowtodepth_cpu<false,true>(
            output.data(),
            flow.data(),
            intrinsics.data(),
            rotation.data(),
            translation.data(),
            flow_shape.dim_size(flow_rank-1),
            flow_shape.dim_size(flow_rank-2),
            w_size );
    }
    else
    {
      if( normalized_flow )
        flowtodepth_cpu<true,false>(
            output.data(),
            flow.data(),
            intrinsics.data(),
            rotation.data(),
            translation.data(),
            flow_shape.dim_size(flow_rank-1),
            flow_shape.dim_size(flow_rank-2),
            w_size );
      else
        flowtodepth_cpu<false,false>(
            output.data(),
            flow.data(),
            intrinsics.data(),
            rotation.data(),
            translation.data(),
            flow_shape.dim_size(flow_rank-1),
            flow_shape.dim_size(flow_rank-2),
            w_size );
    }
    
  }


  template <bool NORMALIZED_FLOW, bool INVERSE_DEPTH>
  void flowtodepth_cpu( 
      T* out, const T* flow, 
      const T* intrinsics,
      const T* rotation,
      const T* translation,
      int x_size, int y_size, int z_size )
  {
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,3,3> Mat3;
    typedef Eigen::Matrix<T,3,4> Mat3x4;
    const int xy_size = x_size*y_size;
    const T inv_x_size = 1.0/x_size;
    const T inv_y_size = 1.0/y_size;
    const int rotation_step = rotation_format_size(rotation_format);

#pragma omp parallel for
    for( int z = 0; z < z_size; ++z )
    {
      Mat3 K( Mat3::Zero() );
      K(0,0) = intrinsics[4*z+0];
      K(1,1) = intrinsics[4*z+1];
      K(0,2) = intrinsics[4*z+2];
      K(1,2) = intrinsics[4*z+3];
      K(2,2) = 1;

      Eigen::Map<const Vec3> t(translation+3*z);
      Mat3 R = convert_to_rotation_matrix(rotation+z*rotation_step, rotation_format);
      
      // compute the fundamental matrix
      Mat3x4 P1;
      P1 << K, Vec3::Zero();
      Mat3x4 P2;
      P2 << R, t;
      P2 = K*P2;
      Mat3 F = computeFundamentalFromCameras(P1,P2);

      std_vector_Matrix3x4<T> Pvec(2);
      Pvec[0] = P1;
      Pvec[1] = P2;

      std_vector_Vector2<T> xvec(2);
      

      T* depthmap = out+z*xy_size;
      const T* flowmap = flow+2*z*xy_size;
#define DEPTH(x,y) depthmap[(y)*x_size+(x)]
#define FLOW(c,x,y) flowmap[(c)*xy_size+(y)*x_size+(x)]
      for( int y = 0; y < y_size; ++y )
      for( int x = 0; x < x_size; ++x )
      {
        Vec2 x1(x+T(0.5), y+T(0.5));
        Vec2 flowvec;
        x1.x() *= inv_x_size;
        x1.y() *= inv_y_size;
        if( NORMALIZED_FLOW )
        {
          flowvec.x() = FLOW(0,x,y);
          flowvec.y() = FLOW(1,x,y);
        }
        else
        {
          flowvec.x() = FLOW(0,x,y)*inv_x_size;
          flowvec.y() = FLOW(1,x,y)*inv_y_size;
        }

        Vec2 x2 = x1 + flowvec;

        // compute the epipolar line in the second image
        Vec3 l = F*x1.homogeneous();
        l /= l.topRows(2).norm(); // normalize

        // compute the closest point on the epipolar line to x2
        T point_line_distance = l.dot(x2.homogeneous());
        Vec2 x2_on_line = x2 - l.topRows(2)*point_line_distance;
        

        xvec[0] = x1;
        xvec[1] = x2;
        Vec3 X = triangulateLinear(Pvec,xvec);
        if( matrix_is_finite(X) && X.z() > 0 )
        {
          if( INVERSE_DEPTH )
            DEPTH(x,y) = 1/X.z();
          else
            DEPTH(x,y) = X.z();
        }
        else
        {
          DEPTH(x,y) = 0;
        }
        

      }
#undef DEPTH
#undef FLOW
    } // z
  }


private:
  RotationFormat rotation_format;
  bool inverse_depth;
  bool normalized_flow;

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("FlowToDepth2")                                                      \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    FlowToDepth2Op<type>);                                                     
REG_KB(float)
REG_KB(double)
#undef REG_KB


