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
#include "helper.h"
#include "cuda_helper.h"

namespace rotation_format_internal
{
  __device__ inline float Sqrt(float x) { return sqrtf(x); }
  __device__ inline double Sqrt(double x) { return sqrt(x); }
  __device__ inline float Cos(float x) { return cosf(x); }
  __device__ inline double Cos(double x) { return cos(x); }
  __device__ inline float Sin(float x) { return sinf(x); }
  __device__ inline double Sin(double x) { return sin(x); }

  template <class T, bool COLUMN_MAJOR>
  __global__ void angleaxis_to_rotmatrix_kernel(T* out, const T* in, int size)
  {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= size )
      return;

    const T* aa = in+3*x;
    T* R = out+9*x;
    T angle = Sqrt(aa[0]*aa[0]+aa[1]*aa[1]+aa[2]*aa[2]);

    if( angle > T(1e-6) )
    {
      T c = Cos(angle);
      T s = Sin(angle);
      T u[3] = {aa[0]/angle, aa[1]/angle, aa[2]/angle};

      if( COLUMN_MAJOR )
      {
        R[0] = c+u[0]*u[0]*(1-c);      R[3] = u[0]*u[1]*(1-c)-u[2]*s; R[6] = u[0]*u[2]*(1-c)+u[1]*s;
        R[1] = u[1]*u[0]*(1-c)+u[2]*s; R[4] = c+u[1]*u[1]*(1-c);      R[7] = u[1]*u[2]*(1-c)-u[0]*s;
        R[2] = u[2]*u[0]*(1-c)-u[1]*s; R[5] = u[2]*u[1]*(1-c)+u[0]*s; R[8] = c+u[2]*u[2]*(1-c);
      }
      else
      {
        R[0] = c+u[0]*u[0]*(1-c);      R[1] = u[0]*u[1]*(1-c)-u[2]*s; R[2] = u[0]*u[2]*(1-c)+u[1]*s;
        R[3] = u[1]*u[0]*(1-c)+u[2]*s; R[4] = c+u[1]*u[1]*(1-c);      R[5] = u[1]*u[2]*(1-c)-u[0]*s;
        R[6] = u[2]*u[0]*(1-c)-u[1]*s; R[7] = u[2]*u[1]*(1-c)+u[0]*s; R[8] = c+u[2]*u[2]*(1-c);
      }
    }
    else
    {
      R[0] = 1; R[3] = 0; R[6] = 0;
      R[1] = 0; R[4] = 1; R[7] = 0;
      R[2] = 0; R[5] = 0; R[8] = 1;
    }
  }

}
using namespace rotation_format_internal;

template <class T>
void angleaxis_to_rotmatrix_gpu(
      const cudaStream_t& stream,
      T* out,
      const T* in,
      int size,
      bool column_major )
{
  dim3 block(128,1,1);
  dim3 grid;
  grid.x = divup(size,block.x);
  grid.y = 1;
  grid.z = 1;

  if( column_major )
    angleaxis_to_rotmatrix_kernel<T,true><<<grid,block,0,stream>>>( out, in, size );
  else
    angleaxis_to_rotmatrix_kernel<T,false><<<grid,block,0,stream>>>( out, in, size );
  CHECK_CUDA_ERROR
}
template void angleaxis_to_rotmatrix_gpu<float>(const cudaStream_t&, float*, const float*, int, bool);
template void angleaxis_to_rotmatrix_gpu<double>(const cudaStream_t&, double*, const double*, int, bool);

