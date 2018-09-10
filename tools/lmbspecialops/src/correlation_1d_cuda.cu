#define EIGEN_USE_GPU
#include "config.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "helper.h"
#include "cuda_helper.h"
#include "Eigen/Core"

#define ROUND_OFF 50000

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

using namespace tensorflow;

// == Dimension rearrangement Kernel
namespace blob_rearrange_kernel2_internal
{
template <class T>
__global__ void blob_rearrange_kernel2_1D(const T* in, T* out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight)
{
    int xy = blockIdx.x*blockDim.x + threadIdx.x;
    if(xy>=widthheight)
        return;

    int ch = blockIdx.y;
    int n  = blockIdx.z;

    T value=in[(n*channels+ch)*widthheight+xy];

    __syncthreads();

    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + 0);
    int xypad = ypad * (width+2*padding) + xpad;

    out[(n*pwidthheight+xypad)*channels + ch] = value;
}
}
using namespace blob_rearrange_kernel2_internal;

// == Correlation Kernel
namespace CorrelateData_internal
{
template <class T>
__global__ void CorrelateData_1D(const int nthreads, int num, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const T *bottom0, const T *bottom1, T *top)
{
  extern __shared__ char patch_data_char[];

  T *patch_data = (T *)patch_data_char;

    // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
  int x1 = blockIdx.x*stride1 + max_displacement;
  int y1 = blockIdx.y*stride1;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;

  // Load 3D patch into shared shared memory
  for(int j = 0; j < kernel_size; j++) { // HEIGHT
    for(int i = 0; i < kernel_size; i++) { // WIDTH
      int ji_off = ((j * kernel_size) + i) * bottomchannels;
      for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = bottom0[idx1];
      }
    }
  }

  __syncthreads();

  __shared__ T sum[WARPS_PER_BLOCK*THREADS_PER_WARP];

  // Compute correlation
  for(int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;

    int s2o = (top_channel % neighborhood_grid_width + x_shift) * stride2;

    for(int j = 0; j < kernel_size; j++) { // HEIGHT
      for(int i = 0; i < kernel_size; i++) { // WIDTH
        int ji_off = ((j * kernel_size) + i) * bottomchannels;
        for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int x2 = x1 + s2o;

          int idxPatchData = ji_off + ch;
          int idx2 = ((item * bottomheight + y1+j) * bottomwidth + x2+i) * bottomchannels + ch;

          sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];
        }
      }
    }

    __syncthreads();

    if(ch_off == 0) {
        T total_sum = 0;
        for(int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
            total_sum += sum[idx];
        }
        const int sumelems = kernel_size*kernel_size*bottomchannels;
        const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
        top[index + item*topcount] = total_sum / (float)sumelems;
    }
  }


  // Aggregate
}
}
using namespace CorrelateData_internal;

// == Correlation Kernel Subtraction
namespace CorrelateDataSubtract_internal
{
template <class T>
__global__ void CorrelateDataSubtract_1D(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const T *bottom0, const T *bottom1, T *top)
{
  LMBOPS_CUDA_KERNEL_LOOP(index, nthreads) {
    int x = index % topwidth; //w-pos
    int y = (index / topwidth) % topheight; //h-pos
    int c = (index / topwidth / topheight) % topchannels; //channels

    // Offset of patch in image 2
    int s2o = (c % neighborhood_grid_width + x_shift) * stride2;

    // First (upper left) position of kernel center in current neighborhood in image 1
    int x1 = x*stride1 + kernel_radius + max_displacement;
    int y1 = y*stride1 + kernel_radius + 0;

    // Iterate through 3D patch
    T sum = 0;
    for(int j = -kernel_radius; j <= kernel_radius; j++) { // HEIGHT
      for(int i = -kernel_radius; i <= kernel_radius; i++) { // WIDTH
        for(int l = 0; l < bottomchannels; l++) { // CHANNELS
          // Calculate position in image 2
          int x2 = x1 + s2o;
          int y2 = y1 ;

          // Indices in bottom data: (CH=l,W=x2,H=y2,N)
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + l;
          int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + l;

          // Do the correlation:
          sum += fabsf(bottom0[idx1] - bottom1[idx2]);
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    top[index + item*topcount] = sum / (float)sumelems;
  }

}
}
using namespace CorrelateDataSubtract_internal;

// == Correlation Backward Pass Kernel (For Blob 0)
namespace CorrelateDataBackward0_internal
{
template <class T>
__global__ void CorrelateDataBackward0_1D(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  T *bottom0diff, const T *bottom1, const T *topdiff)
{
  LMBOPS_CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight; //h-pos

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;

    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1

    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1


    T sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

         {
          for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int idxbot1 = ((item * pbottomheight + m) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            T bot1tmp = bottom1[idxbot1]; // bottom1[l+s2o,m+s2p,n]

            // Index offset for topdiff in following loops:
			int op = (o-x_shift);
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot1tmp;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
	const int bot0index = ((n * bottomheight) + m) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item*bottomcount] = sum / (float)sumelems;
  }

}
}
using namespace CorrelateDataBackward0_internal;

// == Correlation Backward Pass Kernel (For Blob 1)
namespace CorrelateDataBackward1_internal
{
template <class T>
__global__ void CorrelateDataBackward1_1D(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const T *bottom0, T *bottom1diff, const T *topdiff)
{

  LMBOPS_CUDA_KERNEL_LOOP(index, nthreads) {
    //int l = index % bottomwidth + pad_size; //w-pos
    //int m = (index / bottomwidth) % bottomheight + pad_size; //h-pos
    //int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight; //h-pos

    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;

    T sum = 0;
    {
      for(int o = x_shift; o < x_shift+neighborhood_grid_width; o++) {

        int s2o = stride2 * o;

        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - 0 - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1

        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m - 0 - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot0 = ((item * pbottomheight + m) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            T bot0tmp = bottom0[idxbot0]; // bottom1[l+s2o,m+s2p,n]

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); //* neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot0tmp;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
		const int bot1index = ((n * bottomheight) + m) * bottomwidth + (l-pad_size);
		bottom1diff[bot1index + item*bottomcount] = sum / (float)sumelems;
  }

}
}
using namespace CorrelateDataBackward1_internal;

// == Correlation Backward Pass Kernel (For Blob 0)
namespace CorrelateDataBackward0Subtract_internal
{
template <class T>
__global__ void CorrelateDataBackward0Subtract_1D(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  T *bottom0diff, const T *bottom0, const T *bottom1, const T *topdiff)
{
  LMBOPS_CUDA_KERNEL_LOOP(index, nthreads) {
    int l = index % bottomwidth + pad_size; //w-pos
    int m = (index / bottomwidth) % bottomheight; //h-pos
    int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;

    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1

    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1


    T sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        {
		  for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++){

            // Get bottom1 data:
            int s2o = stride2 * o;
            int idxbot = ((item * pbottomheight + m) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            T bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m+s2p,n]
            T bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m+s2p,n]
            T sign = (bot0tmp >= bot1tmp) ? T(1.0) : T(-1.0);

            // Index offset for topdiff in following loops:
			int op = (o-x_shift);
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    bottom0diff[index + item*bottomcount] = sum / (float)sumelems;
  }

}
}
using namespace CorrelateDataBackward0Subtract_internal;

// == Correlation Backward Pass Kernel (For Blob 1)
namespace CorrelateDataBackward1Subtract_internal
{
template <class T>
__global__ void CorrelateDataBackward1Subtract_1D(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const T *bottom0, const T *bottom1, T *bottom1diff, const T *topdiff)
{
  LMBOPS_CUDA_KERNEL_LOOP(index, nthreads) {
    int l = index % bottomwidth + pad_size; //w-pos
    int m = (index / bottomwidth) % bottomheight; //h-pos
    int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels

    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;

    T sum = 0;
    {
      for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++){

        int s2o = stride2 * o;

        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - 0 - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1

        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m -0 - 0  + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot = ((item * pbottomheight + m) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            T bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m+s2p,n]
            T bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m+s2p,n]
            T sign = (bot0tmp >= bot1tmp) ? T(-1.0) : T(1.0);

            // Index offset for topdiff in following loops:
			int op = (o-x_shift); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    bottom1diff[index + item*bottomcount] = sum / (float)sumelems;
  }

}
}
using namespace CorrelateDataBackward1Subtract_internal;

template <class T>
class Correlation1DOp_GPU : public OpKernel
{
public:
  explicit Correlation1DOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction)
  {
    std::string corr_type_str;
    OP_REQUIRES_OK(construction, construction->GetAttr("corr_type", &corr_type_str));
    if( corr_type_str == "mult" )
      corr_type_ = MULT;
    else
      corr_type_ = SUBT;

    // Get attributes
    OP_REQUIRES_OK(construction, construction->GetAttr("kernel_size", &kernel_size_));
    OP_REQUIRES_OK(construction, construction->GetAttr("max_displacement", &max_displacement_));
    OP_REQUIRES_OK(construction, construction->GetAttr("pad_size", &pad_size_));
    OP_REQUIRES_OK(construction, construction->GetAttr("stride1", &stride1_));
    OP_REQUIRES_OK(construction, construction->GetAttr("stride2", &stride2_));
    OP_REQUIRES_OK(construction, construction->GetAttr("do_abs", &do_abs_));
	OP_REQUIRES_OK(construction, construction->GetAttr("single_dir", &single_dir_));

    OP_REQUIRES(construction, kernel_size_ % 2 != 0, errors::InvalidArgument("Correlation cannot be done with even kernel_size"));
  }

  void Compute( OpKernelContext* context ) override
  {
    // Get the inputs
    const Tensor& input1_tensor = context->input(0);
    const Tensor& input2_tensor = context->input(1);

    // Get the shapes
    const TensorShape input1_shape(input1_tensor.shape());
    const TensorShape input2_shape(input2_tensor.shape());

    int num      = input1_shape.dim_size(0);
    int channels = input1_shape.dim_size(1);
    int height   = input1_shape.dim_size(2);
    int width    = input1_shape.dim_size(3);

    int paddedbottomheight = height;
    int paddedbottomwidth = width+2*pad_size_;
	int neighborhood_grid_width, neighborhood_grid_radius;
    int rsize = num*paddedbottomheight*paddedbottomwidth*channels;

    // Size computation
    kernel_radius_ = (kernel_size_ - 1) / 2; //size of unreachable border region (on each side)
    border_size_ = max_displacement_ + kernel_radius_; //size of unreachable border region (on each side)

    top_width_ = ceil((float)(paddedbottomwidth - border_size_*2) / (float)stride1_);
    top_height_ = ceil((float)(paddedbottomheight - kernel_radius_*2) / (float)stride1_);

    OP_REQUIRES(context, top_width_ >= 1, errors::InvalidArgument("Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob"));
    OP_REQUIRES(context, top_height_ >= 1, errors::InvalidArgument("Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob"));

    // Given a center position in image 1, how many displaced positions in -x / +x direction do we consider in image 2 (neighborhoodGridWidth):
    neighborhood_grid_radius_ = max_displacement_ / stride2_;
	if(single_dir_ !=0)
    	neighborhood_grid_width_ = neighborhood_grid_radius_ + 1;
	else
    	neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;


    // Top Channels amount to displacement combinations in X and Y direction:
    top_channels_ = neighborhood_grid_width_ ;

    TensorShape output_shape;
    output_shape.AddDim(num);
    output_shape.AddDim(top_channels_);
    output_shape.AddDim(top_height_);
    output_shape.AddDim(top_width_);

    // Allocate memory for the output image
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    auto device = context->eigen_gpu_device();

    // rbots (These are the blobs that store the padded and dimension rearranged data
    TensorShape rbot_shape;
    rbot_shape.AddDim(num);
    rbot_shape.AddDim(paddedbottomheight);
    rbot_shape.AddDim(paddedbottomwidth);
    rbot_shape.AddDim(channels);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, rbot_shape, &rbot1_gpu_tensor));
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, rbot_shape, &rbot2_gpu_tensor));

    // Prepare for correlation
    auto input1 = input1_tensor.flat<T>();
    auto input2 = input2_tensor.flat<T>();
    auto output = output_tensor->flat<T>();

    auto rbot1 = rbot1_gpu_tensor.flat<T>();
    auto rbot2 = rbot2_gpu_tensor.flat<T>();

	int x_shift = - neighborhood_grid_radius_;
    if(single_dir_ == -1) { // to the left
      x_shift = -neighborhood_grid_width_;
    } else if(single_dir_ == 1) { // to the right
      x_shift = 0;
    }

    Correlation_GPU(device.stream(), output.data(), rbot1.data(), rbot2.data(), input1.data(), input2.data(), num, channels, height, width, rsize, x_shift);
  }

  void Correlation_GPU( const cudaStream_t& stream,
                        T*                  output,
                        T*                  rbot1,
                        T*                  rbot2,
                        const T*            input1,
                        const T*            input2,
                        const int           bnum,
                        const int           bchannels,
                        const int           bheight,
                        const int           bwidth,
                        const int           rsize,
						const int 			x_shift
                      )
  {
    const int bwidthheight = bwidth * bheight;

    const int topcount = top_width_ * top_height_ * top_channels_;

    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);

    cudaMemset(rbot1, 0, rsize*sizeof(T));
    cudaMemset(rbot2, 0, rsize*sizeof(T));

    int threads_per_block=16;
    dim3 totalBlocksRearr((bwidthheight-1)/threads_per_block+1, bchannels, bnum);
    const int pwidthheight = (bwidth + 2 * pad_size_) * (bheight);

    blob_rearrange_kernel2_1D<T><<<totalBlocksRearr,threads_per_block, 0, stream>>>
            (input1,rbot1,bnum,bchannels,bwidth,bheight,bwidthheight,pad_size_,pwidthheight);

    blob_rearrange_kernel2_1D<T><<<totalBlocksRearr,threads_per_block, 0, stream>>>
            (input2,rbot2,bnum,bchannels,bwidth,bheight,bwidthheight,pad_size_,pwidthheight);

    const int num = bnum;
    const int channels = bchannels;
    const int height = bheight;
    const int width = bwidth + 2*pad_size_;

    const int shared_memory_per_block = (kernel_size_*kernel_size_)*bchannels;

    if(corr_type_ == MULT) {
        // CorrelationLayer
        int topThreadCount = topcount;

        dim3 totalBlocksCorr(top_width_, top_height_, num);


        CorrelateData_1D<T><<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(T), stream>>>(
            topThreadCount,
            num, top_width_, top_height_, top_channels_, topcount,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_, kernel_size_,
            stride1_, stride2_,
            width, height, channels,
            rbot1, rbot2, output
            );

        CHECK_CUDA_ERROR;

    } else if(corr_type_ == SUBT) {
        // CorrelationLayer
        for(int n = 0; n < num; n++) {

            int topThreadCount = topcount;

            CorrelateDataSubtract_1D<T><<<LMBOPS_GET_BLOCKS(topThreadCount), LMBOPS_CUDA_NUM_THREADS, 0, stream>>>(
                topThreadCount,
                num, n, top_width_, top_height_, top_channels_, topcount,
                max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
                stride1_, stride2_,
                width, height, channels,
                rbot1, rbot2, output
                );


            CHECK_CUDA_ERROR;
        }
    }
  }
private:
  int kernel_size_;

  int stride1_;
  int stride2_;
  int max_displacement_;

  int pad_size_;

  int num_;
  int top_height_, top_width_;
  int top_channels_;

  // Correlation specific
  bool do_abs_;

  enum CorrType {MULT = 1, SUBT = 2};
  CorrType corr_type_;

  Tensor rbot1_gpu_tensor;
  Tensor rbot2_gpu_tensor;

  // Computed
  int kernel_radius_;
  int border_size_;
  int neighborhood_grid_radius_, neighborhood_grid_width_;
  int single_dir_;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("Correlation1D")                                                       \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    Correlation1DOp_GPU<type>);
REG_KB(float)
#undef REG_KB


template <class T>
class Correlation1DGradOp_GPU : public OpKernel
{
public:
  explicit Correlation1DGradOp_GPU(OpKernelConstruction* construction)
    :OpKernel(construction)
  {
    std::string corr_type_str;
    OP_REQUIRES_OK(construction, construction->GetAttr("corr_type", &corr_type_str));
    if( corr_type_str == "mult" )
      corr_type_ = MULT;
    else
      corr_type_ = SUBT;

    // Get attributes
    OP_REQUIRES_OK(construction, construction->GetAttr("kernel_size", &kernel_size_));
    OP_REQUIRES_OK(construction, construction->GetAttr("max_displacement", &max_displacement_));
    OP_REQUIRES_OK(construction, construction->GetAttr("pad_size", &pad_size_));
    OP_REQUIRES_OK(construction, construction->GetAttr("stride1", &stride1_));
    OP_REQUIRES_OK(construction, construction->GetAttr("stride2", &stride2_));
    OP_REQUIRES_OK(construction, construction->GetAttr("do_abs", &do_abs_));
    OP_REQUIRES_OK(construction, construction->GetAttr("single_dir", &single_dir_));

    OP_REQUIRES(construction, kernel_size_ % 2 != 0, errors::InvalidArgument("Correlation cannot be done with even kernel_size"));
  }

  void Compute( OpKernelContext* context ) override
  {
    // Get the inputs
    const Tensor& input1_tensor   = context->input(0);
    const Tensor& input2_tensor   = context->input(1);
    const Tensor& gradient_tensor = context->input(2);

    // Get the shapes
    const TensorShape input1_shape(input1_tensor.shape());
    const TensorShape input2_shape(input2_tensor.shape());

    int num      = input1_shape.dim_size(0);
    int channels = input1_shape.dim_size(1);
    int height   = input1_shape.dim_size(2);
    int width    = input1_shape.dim_size(3);

    int paddedbottomheight = height;
    int paddedbottomwidth = width+2*pad_size_;

    int rsize = num*paddedbottomheight*paddedbottomwidth*channels;

    // Size computation
    kernel_radius_ = (kernel_size_ - 1) / 2; //size of unreachable border region (on each side)
    border_size_ = max_displacement_ + kernel_radius_; //size of unreachable border region (on each side)

    top_width_ = ceil((float)(paddedbottomwidth - border_size_*2) / (float)stride1_);
    top_height_ = ceil((float)(paddedbottomheight - kernel_radius_*2) / (float)stride1_);

    OP_REQUIRES(context, top_width_ >= 1, errors::InvalidArgument("Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob"));
    OP_REQUIRES(context, top_height_ >= 1, errors::InvalidArgument("Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob"));

    // Given a center position in image 1, how many displaced positions in -x / +x direction do we consider in image 2 (neighborhoodGridWidth):
    neighborhood_grid_radius_ = max_displacement_ / stride2_;
 	if(single_dir_ != 0) {
		neighborhood_grid_width_ = neighborhood_grid_radius_ + 1;
  	} else {
    	neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
  	}

    // Top Channels amount to displacement combinations in X and Y direction:
    top_channels_ = neighborhood_grid_width_ ;

    // Allocate the memory for the output
    Tensor* input1_grad_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, input1_shape, &input1_grad_tensor));
    Tensor* input2_grad_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, input2_shape, &input2_grad_tensor));

    auto device = context->eigen_gpu_device();

    // rbots (These are the blobs that store the padded and dimension rearranged data
    TensorShape rbot_shape;
    rbot_shape.AddDim(num);
    rbot_shape.AddDim(paddedbottomheight);
    rbot_shape.AddDim(paddedbottomwidth);
    rbot_shape.AddDim(channels);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, rbot_shape, &rbot1_gpu_tensor));
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, rbot_shape, &rbot2_gpu_tensor));

    // Prepare for gradient computation
    auto input1      = input1_tensor.flat<T>();
    auto input2      = input2_tensor.flat<T>();
    auto gradient    = gradient_tensor.flat<T>();
    auto input1_grad = input1_grad_tensor->flat<T>();
    auto input2_grad = input2_grad_tensor->flat<T>();

    auto rbot1 = rbot1_gpu_tensor.flat<T>();
    auto rbot2 = rbot2_gpu_tensor.flat<T>();

	int x_shift = - neighborhood_grid_radius_ ;

    if(single_dir_ == -1) { // to the left
      x_shift = -neighborhood_grid_width_;
    } else if(single_dir_ == 1) { // to the right
      x_shift = 0;
    }

    CorrelationGrad_GPU(device.stream(), input1_grad.data(), input2_grad.data(), rbot1.data(), rbot2.data(), input1.data(), input2.data(), gradient.data(), num, channels, height, width, paddedbottomheight, paddedbottomwidth, rsize, x_shift);

  	}


  void CorrelationGrad_GPU( const cudaStream_t& stream,
                            T*                  input1_grad,
                            T*                  input2_grad,
                            T*                  rbot1,
                            T*                  rbot2,
                            const T*            input1,
                            const T*            input2,
                            const T*            gradient,
                            const int           bnum,
                            const int           bchannels,
                            const int           bheight,
                            const int           bwidth,
                            const int           paddedheight,
                            const int           paddedwidth,
                            const int           rsize,
							const int 			x_shift
                          )
  {
    const int bwidthheight = bwidth * bheight;

    cudaMemset(rbot1, 0, rsize*sizeof(T));
    cudaMemset(rbot2, 0, rsize*sizeof(T));

    int threads_per_block=16;
    dim3 totalBlocksRearr((bwidthheight-1)/threads_per_block+1, bchannels, bnum);
    const int pwidthheight = (bwidth + 2 * pad_size_) * (bheight);

    blob_rearrange_kernel2_1D<T><<<totalBlocksRearr,threads_per_block, 0, stream>>>
            (input1,rbot1,bnum,bchannels,bwidth,bheight,bwidthheight,pad_size_,pwidthheight);
    CHECK_CUDA_ERROR;
    blob_rearrange_kernel2_1D<T><<<totalBlocksRearr,threads_per_block, 0, stream>>>
            (input2,rbot2,bnum,bchannels,bwidth,bheight,bwidthheight,pad_size_,pwidthheight);
    CHECK_CUDA_ERROR;

    const int num = bnum;
    const int channels = bchannels;
    const int height = bheight;
    const int width = bwidth;

    const int bottomcount = channels * height * width;

    int botThreadCount = bottomcount;

    // CorrelationLayerBackward
    if(corr_type_ == MULT) {

        // == Run kernel Backward 0
        dim3 totalBlocksBackward0(width, height, channels * num); //First dim is fastest
        dim3 threadsPerBlockBackward0(THREADS_PER_WARP * WARPS_PER_BLOCK);
        for(int n = 0; n < num; n++) {
        //Bottom0:
        CorrelateDataBackward0_1D<T><<<LMBOPS_GET_BLOCKS(botThreadCount), LMBOPS_CUDA_NUM_THREADS, 0, stream>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            input1_grad, rbot2, gradient
            );

        CHECK_CUDA_ERROR;
        }

        // == Run kernel Backward 1
        for(int n = 0; n < num; n++) {
        CorrelateDataBackward1_1D<T><<<LMBOPS_GET_BLOCKS(botThreadCount), LMBOPS_CUDA_NUM_THREADS, 0, stream>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            rbot1, input2_grad, gradient
            );

        CHECK_CUDA_ERROR;
        }

    } else if(corr_type_ == SUBT) {
        for(int n = 0; n < num; n++) {
        //Bottom0:
        CorrelateDataBackward0Subtract_1D<T><<<LMBOPS_GET_BLOCKS(botThreadCount), LMBOPS_CUDA_NUM_THREADS, 0, stream>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            input1_grad, rbot1, rbot2, gradient
            );

        CHECK_CUDA_ERROR;
        }

        for(int n = 0; n < num; n++) {
        //Bottom0:
        CorrelateDataBackward1Subtract_1D<T><<<LMBOPS_GET_BLOCKS(botThreadCount), LMBOPS_CUDA_NUM_THREADS, 0, stream>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            rbot1, rbot2, input2_grad, gradient
            );

        CHECK_CUDA_ERROR;
        }
    }
  }
private:
  int kernel_size_;

  int stride1_;
  int stride2_;
  int max_displacement_;

  int pad_size_;

  int num_;
  int top_height_, top_width_;
  int top_channels_;

  // Correlation specific
  bool do_abs_;

  enum CorrType {MULT = 1, SUBT = 2};
  CorrType corr_type_;

  Tensor rbot1_gpu_tensor;
  Tensor rbot2_gpu_tensor;

  // Computed
  int kernel_radius_;
  int border_size_;
  int neighborhood_grid_radius_, neighborhood_grid_width_;
  int single_dir_;

};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("Correlation1DGrad")                                                   \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    Correlation1DGradOp_GPU<type>);
REG_KB(float)
#undef REG_KB
