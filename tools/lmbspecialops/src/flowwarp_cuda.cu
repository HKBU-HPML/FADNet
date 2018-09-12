//
//  lmbspecialops - a collection of tensorflow ops
//  Copyright (C) 2017  Oezguen Cicek, Eddy Ilg
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

#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))

#define RA_TILE 32
#define RA_ROWS 8

#define FW_THREADS 32
#define FW_TILE_X FW_THREADS
#define FW_TILE_C FW_THREADS

#define ZERO 1
#define NOT_A_NUMBER 2

namespace flow_warp_rearrange_kernel_internal
{
template <class T>
__global__ void flow_warp_rearrange_kernel(const T* in, T* out, int num, int channels, int cblocks, int width, int height, int widthheight)
{
  __shared__ float buffer[RA_TILE][RA_TILE+1];

  int n  = blockIdx.x/cblocks;
  if(n>=num) return;

  int c0 = (blockIdx.x%cblocks)*RA_TILE;
  int x0 = blockIdx.y*RA_TILE;
  int y  = blockIdx.z;

  int xoff=threadIdx.x;
  int coff=threadIdx.y;
  int x=x0+xoff;

  if(x<width)
    for(int i=coff; i<RA_TILE && c0+i<channels; i+=RA_ROWS)
        buffer[i][xoff] = in[((n*channels + c0 + i)*height + y)*width + x];

  __syncthreads();

  coff = threadIdx.x;
  xoff = threadIdx.y;
  int c = c0 + coff;

  if(c<channels)
    for(int j=xoff; j<RA_TILE && x0+j<width; j+=RA_ROWS)
       out[((n*height + y)*width + x0+j)*channels + c] = buffer[coff][j];
}
}
using namespace flow_warp_rearrange_kernel_internal;

namespace flow_warp_kernel_smem_internal
{
template <class T>
__global__ void flow_warp_kernel_smem(const T* image, const T* flow, T* warped, int num, int channels, int cblocks, int width, int wblocks, int height, int widthheight, float fillValue)
{
    int y = blockIdx.y;
    int n = blockIdx.z;

    __shared__ float x2_buf[FW_TILE_X], y2_buf[FW_TILE_X];
    __shared__ float buffer[FW_TILE_C][FW_TILE_X+1];

    int x;
    int c;

    x = blockIdx.x*FW_TILE_X + threadIdx.x;
    if(threadIdx.y==0 && x<width)
    {
        x2_buf[threadIdx.x] = float(x) + flow[((2*n  )*height + y)*width + x];
        y2_buf[threadIdx.x] = float(y) + flow[((2*n+1)*height + y)*width + x];
    }

    __syncthreads();

    float x2 = x2_buf[threadIdx.y];
    float y2 = y2_buf[threadIdx.y];

    int ix2_L = int(x2);
    int iy2_T = int(y2);
    int ix2_R = min(ix2_L+1, width-1);
    int iy2_B = min(iy2_T+1, height-1);

    int off_TL = ((n*height + iy2_T)*width + ix2_L)*channels;
    int off_TR = ((n*height + iy2_T)*width + ix2_R)*channels;
    int off_BL = ((n*height + iy2_B)*width + ix2_L)*channels;
    int off_BR = ((n*height + iy2_B)*width + ix2_R)*channels;

    float alpha = x2-ix2_L;
    float beta = y2-iy2_T;
    float coeffTL = (1-alpha)*(1-beta);
    float coeffTR = alpha*(1-beta);
    float coeffBL = (1-alpha)*beta;
    float coeffBR = alpha*beta;

    for(int cb=0; cb<cblocks; cb++)
    {
        __syncthreads();

        buffer[threadIdx.y][threadIdx.x] = fillValue;

        __syncthreads();

        c = cb*FW_TILE_C + threadIdx.x;
        if(x2>=0 && y2>=0 && x2<width && y2<height && c<channels)
            buffer[threadIdx.y][threadIdx.x] =  // buffer [x][c]
                coeffTL * image[off_TL + c] +
                coeffTR * image[off_TR + c] +
                coeffBL * image[off_BL + c] +
                coeffBR * image[off_BR + c];

        __syncthreads();

        c = cb*FW_TILE_C + threadIdx.y;
        x = blockIdx.x*FW_TILE_X + threadIdx.x;
        if(c<channels && x<width)
            warped[((n*channels+c)*height + y)*width + x] = buffer[threadIdx.x][threadIdx.y];
    }
}
}
using namespace flow_warp_kernel_smem_internal;

namespace flow_warp_backward_kernel_no_smem_internal
{
template <class T>
__global__ void flow_warp_backward_kernel_no_smem(
        const T* image_data, T* image_diff, const T* flow_data, T* flow_diff, const T* warped_diff,
        int num, int channels, int cblocks, int width, int wblocks, int height, int widthheight)
{
    int x = blockIdx.x*FW_TILE_X + threadIdx.x;
    if(x>=width)
        return;

    int y = blockIdx.y;
    int n = blockIdx.z;

    float x2 = float(x) + flow_data[((2*n  )*height + y)*width + x];
    float y2 = float(y) + flow_data[((2*n+1)*height + y)*width + x];

    if(x2>=0.f && y2>=0.f && x2<width && y2<height)
    {
        int ix2_L = int(x2);
        int iy2_T = int(y2);
        int ix2_R = min(ix2_L+1, width-1);
        int iy2_B = min(iy2_T+1, height-1);

        float alpha=x2-ix2_L;
        float beta=y2-iy2_T;
        for(int c=0; c<channels; c++)
        {
            int ch_off = (n*channels + c)*height;
            float warped_diff_value = warped_diff[(ch_off + y)*width + x];
            atomicAdd(&image_diff[(ch_off + iy2_T)*width + ix2_L], warped_diff_value * (1-alpha)*(1-beta));
            atomicAdd(&image_diff[(ch_off + iy2_T)*width + ix2_R], warped_diff_value * alpha*(1-beta));
            atomicAdd(&image_diff[(ch_off + iy2_B)*width + ix2_L], warped_diff_value * (1-alpha)*beta);
            atomicAdd(&image_diff[(ch_off + iy2_B)*width + ix2_R], warped_diff_value * alpha*beta);
        }

        float gamma = iy2_B - y2;
        float bot_diff = 0;
        for(int c=0; c<channels; c++)
        {
            int ch_off = (n*channels + c)*height;
            float temp = 0;
            temp += gamma *     (image_data[(ch_off + iy2_T)*width + ix2_R] - image_data[(ch_off + iy2_T)*width + ix2_L]);
            temp += (1-gamma) * (image_data[(ch_off + iy2_B)*width + ix2_R] - image_data[(ch_off + iy2_B)*width + ix2_L]);

            bot_diff += warped_diff[(ch_off + y)*width + x] * temp;
        }
        flow_diff[(2*n*height + y)*width + x] = bot_diff;

        gamma = ix2_R - x2;
        bot_diff = 0;
        for(int c=0; c<channels; c++)
        {
            int ch_off = (n*channels + c)*height;
            float temp = 0;
            temp += gamma *     (image_data[(ch_off + iy2_B)*width + ix2_L] - image_data[(ch_off + iy2_T)*width + ix2_L]);
            temp += (1-gamma) * (image_data[(ch_off + iy2_B)*width + ix2_R] - image_data[(ch_off + iy2_T)*width + ix2_R]);

            bot_diff += warped_diff[(ch_off + y)*width + x] * temp;
        }
        flow_diff[((2*n+1)*height + y)*width + x] = bot_diff;
    }
}
}
using namespace flow_warp_backward_kernel_no_smem_internal;

template <class T>
class FlowWarpOp_GPU : public OpKernel 
{
public:
  explicit FlowWarpOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction) 
  {
    std::string fill_parameter_str;
    OP_REQUIRES_OK(construction, construction->GetAttr("fill_parameter", &fill_parameter_str));
    if( fill_parameter_str == "zero" )
      fill_parameter = ZERO;
    else 
      fill_parameter = NOT_A_NUMBER;
  }
  
  void Compute( OpKernelContext* context ) override 
  {
    // Get the inputs
    const Tensor& image_tensor = context->input(0);
    const Tensor& flow_tensor  = context->input(1);
    
    // Get the shapes
    const TensorShape image_shape(image_tensor.shape());
    
    // Allocate memory for the warped image
    Tensor* warped_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, image_shape, &warped_tensor));
    
    // Prepare for warping
    auto image = image_tensor.flat<T>();
    auto flow  = flow_tensor.flat<T>();
    auto warped = warped_tensor->flat<T>();
    
    int num      = image_shape.dim_size(0);
    int channels = image_shape.dim_size(1);
    int height   = image_shape.dim_size(2);
    int width    = image_shape.dim_size(3);
    
    // Prepare transposed image
    TensorShape transposed_image_shape;
    transposed_image_shape.AddDim(num);
    transposed_image_shape.AddDim(height);
    transposed_image_shape.AddDim(width);
    transposed_image_shape.AddDim(channels);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, transposed_image_shape, &transposed_image_tensor));
    auto trans_image =  transposed_image_tensor.flat<T>();
    
    auto device = context->eigen_gpu_device();
    
    int nan = 0xFFE00000;
    float nanf = *(reinterpret_cast<float*>(&nan));
    float fill_value = fill_parameter == ZERO ? 0 : nanf;
    
    FlowWarp_GPU(device.stream(), warped.data(), trans_image.data(), image.data(), flow.data(), fill_value, num, channels, height, width);
  }
  
  void FlowWarp_GPU(const cudaStream_t& stream,
                    T*                  warped, 
                    T*                  trans_image, 
                    const T*            image, 
                    const T*            flow,
                    const float         fill_value,
                    const int           num, 
                    const int           channels, 
                    const int           height, 
                    const int           width
                   )
  {
    const int wh_size = width * height;
    const int whc_size = width * height * channels; 
    
    cudaMemset(warped, fill_value, width*height*channels*num*sizeof(T));
    
    #ifdef DISPLAY_TIMINGS
      caffe::Timer t1;
      t1.Start();
    #endif
      
    dim3 rearrangeThreads(RA_TILE, RA_ROWS, 1);
    int cblocks = ((channels-1)/RA_TILE+1);
    dim3 rearrangeBlocks(cblocks*num, (width-1)/RA_TILE+1, height);
    
    flow_warp_rearrange_kernel<T><<<rearrangeBlocks, rearrangeThreads, 0, stream>>>(
        image,
        trans_image,
        num,
        channels,
        cblocks,
        width,
        height,
        wh_size
    );

    CHECK_CUDA_ERROR;
    
    #ifdef DISPLAY_TIMINGS
    t1.Stop();
    LOG(INFO) << "rearrange time " << t1.MilliSeconds() << "ms";
    #endif

    {
    #ifdef DISPLAY_TIMINGS
      caffe::Timer t2;
      t2.Start();
    #endif
    int wblocks = ((width-1)/FW_TILE_X+1);
    int cblocks = ((channels-1)/FW_TILE_C+1);
    dim3 warpThreads(FW_TILE_X, FW_TILE_C);
    dim3 warpBlocks(wblocks, height, num);
    
    flow_warp_kernel_smem<T><<<warpBlocks, warpThreads, 0, stream>>>(
        trans_image,
        flow,
        warped,
        num,
        channels,
        cblocks,
        width,
        wblocks,
        height,
        wh_size,
        fill_value
        );
    
    CHECK_CUDA_ERROR;
    
    #ifdef DISPLAY_TIMINGS
      t2.Stop();
      LOG(INFO) << "warp time 1a: " << t2.MilliSeconds() << "ms";
    #endif
    }
  }
private:
  int fill_parameter;
  Tensor  transposed_image_tensor;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("FlowWarp")                                                          \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    FlowWarpOp_GPU<type>);                                                    
REG_KB(float)
#undef REG_KB


template <class T>
class FlowWarpGradOp_GPU : public OpKernel 
{
public:
  explicit FlowWarpGradOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction) {}
  
  void Compute( OpKernelContext* context ) override 
  {
    // Get the inputs   
    const Tensor& image_tensor     = context->input(0);
    const Tensor& flow_tensor      = context->input(1); 
    const Tensor& gradient_tensor  = context->input(2);
    
    // Get the shapes
    const TensorShape image_shape(image_tensor.shape());
    const TensorShape flow_shape(flow_tensor.shape());
    
    // Allocate the memory for the output
    Tensor* image_grad_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, image_shape, &image_grad_tensor));
    Tensor* flow_grad_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, flow_shape,  &flow_grad_tensor));
    
    // Prepare for gradient computation    
    auto flow        = flow_tensor.flat<T>(); 
    auto image       = image_tensor.flat<T>();
    auto gradient    = gradient_tensor.flat<T>();
    auto image_grad  = image_grad_tensor->flat<T>();
    auto flow_grad   = flow_grad_tensor->flat<T>();

    int num      = image_shape.dim_size(0);
    int channels = image_shape.dim_size(1);
    int height   = image_shape.dim_size(2);
    int width    = image_shape.dim_size(3);
    
    auto device = context->eigen_gpu_device();
    
    FlowWarpGrad_GPU(device.stream(), image_grad.data(), flow_grad.data(), image.data(), flow.data(), gradient.data(), num, channels, height, width);
  }
  
  void FlowWarpGrad_GPU( const cudaStream_t& stream,
                         T*                  image_grad,
                         T*                  flow_grad,
                         const T*            image,
                         const T*            flow,
                         const T*            gradient,
                         const int           num,
                         const int           channels,
                         const int           height,
                         const int           width
                       ) 
  {
    const int wh_size = width * height;
    const int whc_size = width * height * channels;  
    
    cudaMemset(image_grad, 0, width*height*channels*num*sizeof(float));
    cudaMemset(flow_grad, 0, width*height*2*num*sizeof(float));
    
    #ifdef DISPLAY_TIMINGS
      caffe::Timer t3a;
      t3a.Start();
    #endif
      
    int wblocks = ((width-1)/FW_TILE_X+1);
    int cblocks = ((channels-1)/FW_TILE_C+1);
    dim3 warpThreads(FW_TILE_X,1);
    dim3 warpBlocks(wblocks, height, num);
    
    flow_warp_backward_kernel_no_smem<T><<<warpBlocks, warpThreads, 0, stream>>>(
        image,
        image_grad,
        flow,
        flow_grad,
        gradient,
        num,
        channels,
        cblocks,
        width,
        wblocks,
        height,
        wh_size
    );
    
    CHECK_CUDA_ERROR;

    #ifdef DISPLAY_TIMINGS
      t3a.Stop();
      LOG(INFO) << "backward time 1a: " << t3a.MilliSeconds() << "ms";
    #endif
  }
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("FlowWarpGrad")                                                      \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    FlowWarpGradOp_GPU<type>);                                               
REG_KB(float)
#undef REG_KB