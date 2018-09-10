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
#ifndef CUDA_HELPER_H_
#define CUDA_HELPER_H_
#include <iostream>
#include <stdexcept>

#define CHECK_CUDA_ERROR _CHECK_CUDA_ERROR(__FILE__, __LINE__);

inline void _CHECK_CUDA_ERROR( const char* filename, const int line )
{
  cudaError_t error = cudaGetLastError();
  if( error != cudaSuccess )
  {
    char str[1024]; str[0] = 0;
    sprintf(str, "%s:%d: cuda error: %s\n",
           filename, line, cudaGetErrorString(error));
    throw std::runtime_error(str);
  }
}

inline void print_cuda_pointer_attributes( const void* ptr )
{
  cudaPointerAttributes attr = cudaPointerAttributes();
  cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
  char str[1024]; str[0] = 0;
  if( status != cudaSuccess )
  {
    sprintf(str, "cuda error in 'print_cuda_pointer_attributes()': %s",
        cudaGetErrorString(status));
    //throw std::runtime_error(str);
  }
  std::cerr << "\n" << ptr << "  " << str << "\n";
  std::cerr << "memoryType: "
    << (attr.memoryType==cudaMemoryTypeHost ? "cudaMemoryTypeHost\n" : "cudaMemoryTypeDevice\n")
    << "device: " << attr.device << "\n"
    << "devicePointer: " << attr.devicePointer << "\n"
    << "hostPointer: " << attr.hostPointer << "\n"
    << "isManaged: " << attr.isManaged << "\n";
}
#define LMBOPS_CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int LMBOPS_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int LMBOPS_GET_BLOCKS(const int N) {
  return (N + LMBOPS_CUDA_NUM_THREADS - 1) /LMBOPS_CUDA_NUM_THREADS;
}



#endif /* CUDA_HELPER_H_ */
