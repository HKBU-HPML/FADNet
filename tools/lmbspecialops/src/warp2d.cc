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

REGISTER_OP("Warp2d")
  .Attr("T: {float, double}")
  .Attr("normalized: bool = false")
  .Attr("border_mode: {'clamp', 'value'} = 'clamp'")
  .Attr("border_value: float = 0.0")
  .Input("input: T")
  .Input("displacements: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle input_shape, displacements_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 3, &displacements_shape));

      if( c->RankKnown(displacements_shape) )
      {
        // check if C == 2 for the displacements
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(displacements_shape,-3), 2, &d));
      }


      if( c->RankKnown(input_shape) && c->RankKnown(displacements_shape))
      {
        // check if width and height are compatible
        ShapeHandle input_shape2d;
        c->Subshape(input_shape, -2, &input_shape2d);
        ShapeHandle displacements_shape2d;
        c->Subshape(displacements_shape, -2, &displacements_shape2d);

        ShapeHandle merged_shape2d;
        TF_RETURN_IF_ERROR(c->Merge(input_shape2d, displacements_shape2d, &merged_shape2d));


        int displacements_rank = c->Rank(displacements_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < displacements_rank-3; ++i )
        {
          c->Multiply(first_dim, c->Dim(input_shape,i), &first_dim);
        }

        // check if N is compatible for both inputs
        int input_rank = c->Rank(displacements_shape);
        DimensionHandle input_first_dim = c->MakeDim(1);
        if( input_rank >= 4)
        {
          for( int i = 0; i < input_rank-3; ++i )
          {
            c->Multiply(input_first_dim, c->Dim(input_shape,i), &input_first_dim);
          }
        }
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->Merge(input_first_dim, first_dim, &d));
      }
      c->set_output(0,c->input(0));
      return Status::OK();
    })
  .Doc(R"doc(
Warps the input with the given displacement vector field.



normalized: 
  If true then the displacement vectors are normalized with the width and height of the input.

border_mode:
  Defines how to handle values outside of the image.
  'clamp': Coordinates will be clamped to the valid range.
  'value' : Uses 'border_value' outside the image borders.

border_value:
  The value used outside the image borders.

input:
  Input tensor in the format NCHW with a minimum rank of 2.
  For rank 2 tensors C == 1 is assumed.
  For rank 3 tensors N == 1 is assumed.

displacements: 
  The tensor storing the displacement vector field.
  The format is NCHW with C=2 and the rank is at least 3.
  The first channel is the displacement in x direction (width).
  The second channel is the displacement in y direction (height).

output:
  The warped input tensor.

)doc");



template <class T>
class Warp2dOp : public OpKernel 
{
public:
  explicit Warp2dOp(OpKernelConstruction* construction)
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


    warp2d_cpu(
        output.data(),
        input.data(),
        displacements.data(),
        x_size, y_size, z_size, w_size );
    
  }

  void warp2d_cpu(
      T* out, const T* in, const T* displacements,
      int x_size, int y_size, int z_size, int w_size)
  {
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<int,2,1> Vec2i;
    typedef Eigen::Matrix<T,4,1> Vec4;
    const int xy_size = x_size*y_size;
    const int xyz_size = xy_size*z_size;
#define IN(w,z,y,x) in[(w)*xyz_size+(z)*xy_size+(y)*x_size+(x)]
#define OUT(w,z,y,x) out[(w)*xyz_size+(z)*xy_size+(y)*x_size+(x)]
#define VECTOR(w,z,y,x) displacements[(w)*2*xy_size+(z)*xy_size+(y)*x_size+(x)]
    for( int w = 0; w < w_size; ++w )
    {
      for( int y = 0; y < y_size; ++y )
      for( int x = 0; x < x_size; ++x )
      {
        Vec2 p1(x,y);
        Vec2 v(VECTOR(w,0,y,x), VECTOR(w,1,y,x));
        if( normalized )
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

        if( border_mode == CLAMP )
        {
          int x0, y0, x1, y1, x2, y2, x3, y3;
          x0 = std::min(x_size-1,std::max(0,p2i.x()));
          y0 = std::min(y_size-1,std::max(0,p2i.y()));
          x1 = std::min(x_size-1,std::max(0,p2i.x()+1));
          y1 = std::min(y_size-1,std::max(0,p2i.y()));
          x2 = std::min(x_size-1,std::max(0,p2i.x()));
          y2 = std::min(y_size-1,std::max(0,p2i.y()+1));
          x3 = std::min(x_size-1,std::max(0,p2i.x()+1));
          y3 = std::min(y_size-1,std::max(0,p2i.y()+1));
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

      }
    }
#undef IN
#undef OUT
#undef VECTOR
  }

private:
  enum BorderMode {CLAMP = 1, VALUE = 2};
  BorderMode border_mode;
  T border_value;
  bool normalized;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("Warp2d")                                                            \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    Warp2dOp<type>);                                                    
REG_KB(float)
REG_KB(double)
#undef REG_KB


