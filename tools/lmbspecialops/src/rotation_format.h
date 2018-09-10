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
#ifndef ROTATION_FORMAT_H
#define ROTATION_FORMAT_H
#include "config.h"
#include "Eigen/Geometry"

enum RotationFormat {MATRIX, QUATERNION, ANGLEAXIS3};


inline int rotation_format_size(RotationFormat format)
{
  switch(format)
  {
  case MATRIX: return 9;
  case QUATERNION: return 4;
  case ANGLEAXIS3:
  default: return 3;
  }
}


template <class T>
inline Eigen::Matrix<T,3,3> convert_to_rotation_matrix(const T* data, RotationFormat format)
{
  typedef Eigen::Matrix<T,3,1> Vec3;
  typedef Eigen::Matrix<T,3,3> Mat3;
  typedef Eigen::Quaternion<T> Quaternion;
  typedef Eigen::AngleAxis<T> AngleAxis;

  Eigen::Matrix<T,3,3> R;
  switch(format)
  {
  case MATRIX:
    R = Eigen::Map<const Mat3>(data).transpose(); // Eigen uses column major
    break;
  case QUATERNION:
    {
      Quaternion q;
      q.w() = data[0];
      q.x() = data[1];
      q.y() = data[2];
      q.z() = data[3];

      q.normalize();
      R = q.toRotationMatrix();
    }
    break;
  case ANGLEAXIS3:
    {
      Vec3 axis(
          data[0],
          data[1],
          data[2]);
      T angle = axis.norm();
      if( angle > 1.0e-6 )
      {
        axis /= angle;
        R = AngleAxis(angle, axis).toRotationMatrix();
      }
      else
        R.setIdentity();
    }
    break;
  } //switch
  return R;
}

//
// Converts a 3d angle axis vector to a rotation matrix
//
#ifdef BUILD_WITH_CUDA
#include <cuda_runtime.h>
template <class T> 
void angleaxis_to_rotmatrix_gpu( const cudaStream_t& stream, T* out, const T* in, int size, bool column_major=true );
#endif

#endif /* ROTATION_FORMAT_H */
