#define EIGEN_USE_GPU
#include "config.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "helper.h"
#include "cuda_helper.h"

#define FILTER_BICUBIC 0
#define FILTER_BOX 1
#define FILTER_TRIANGLE 2


using namespace tensorflow;

namespace resample_kernel_internal
{
	static __device__ __forceinline__ float bicubicCoeff(float x_)
	{
		float x = fabsf(x_);
		if (x <= 1.0f)     return x * x * (1.5f * x - 2.5f) + 1.0f;
		else if (x < 2.0f) return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
		else               return 0.0f;
	}

	static __device__ __forceinline__ float boxCoeff(float x)
	{
		if (-0.5 <= x  && x<0.5) return 1.0;
		return 0;
	}

	static __device__ __forceinline__ float triangleCoeff(float x)
	{
		if (-1<=x && x<0) return x+1;
		if (0<=x && x<=1) return 1-x;
		return 0;
	}

	template <typename T>
	__global__ void NearestNeighborKernel(
			const int nthreads,
			const int in_channelsize,
			const int out_channelsize,
			const T* in_ptr,
			const int in_width,
			const int in_height,
			const float fx,
			const float fy,
			T* out_ptr,
			const int out_width,
			const int out_height)
	{
		LMBOPS_CUDA_KERNEL_LOOP(index, nthreads)
		{
			int c = index / out_channelsize;
			int x_out = (index % out_channelsize) % out_width;
			int y_out = (index % out_channelsize) / out_width;

			float x_in = x_out * fx + fy / 2.0f - 0.5f;
			float y_in = y_out * fy + fx / 2.0f - 0.5f;

			int x_in_round = round(x_in);
			int y_in_round = round(y_in);

			out_ptr[index] = in_ptr[c*in_channelsize + y_in_round*in_width+x_in_round];
		}
	}
	template <typename T>
		__global__ void InterpolationKernel(
				const int nthreads,
				const int in_channelsize,
				const int out_channelsize,
				const T* in_ptr,
				const int in_width,
				const int in_height,
				const float fx,
				const float fy,
				T* out_ptr,
				const int out_width,
				const int out_height,
				int filter_type,
				int kernel_width,
				const bool antialias)
		{
			LMBOPS_CUDA_KERNEL_LOOP(index, nthreads)
			{
				int c = index / out_channelsize;
				int x_out = (index % out_channelsize) % out_width;
				int y_out = (index % out_channelsize) / out_width;

				float x_in = x_out * fx + fy / 2.0f - 0.5f;
				float y_in = y_out * fy + fx / 2.0f - 0.5f;

				int x_in_round = round(x_in);
				int y_in_round = round(y_in);

				T sum=0;
				T wsum=0;

				float ax = 1.0f / (antialias ? fx : 1.0f);
				float ay = 1.0f / (antialias ? fy : 1.0f);
				int rx = (fx < 1.0f) ? 2 : ceil(float(kernel_width)/ax);
				int ry = (fy < 1.0f) ? 2 : ceil(float(kernel_width)/ay);

				for(int y=y_in_round-ry; y<=y_in_round+ry; y++)
					for(int x=x_in_round-rx; x<=x_in_round+rx; x++)
					{
						if(y<0 || x<0) continue;
						if(y>=in_height || x>=in_width) continue;

						float dx = x_in - x;
						float dy = y_in - y;

						float w;
						if(filter_type == FILTER_BICUBIC)   w = ax*bicubicCoeff(ax*dx) * ay*bicubicCoeff(ay*dy);
						else if(filter_type == FILTER_BOX)  w = ax*boxCoeff(ax*dx) * ay*boxCoeff(ay*dy);
						else                                w = ax*triangleCoeff(ax*dx) * ay*triangleCoeff(ay*dy);
						sum += w * in_ptr[c*in_channelsize + y*in_width+x];
						wsum += w;
					}

				out_ptr[index] = (!wsum) ? 0 : (sum / wsum);
			}
		}

}


using namespace resample_kernel_internal;

template <class T>
class ResampleOp_GPU: public OpKernel
{
	int width;
	int height;
	enum InterpolationType {NEAREST = 1, CUBIC = 2, LINEAR = 3};
    InterpolationType type;
	bool antialias;

public:
  explicit ResampleOp_GPU(OpKernelConstruction* context)
    :OpKernel(context)
  {
	std::string type_str;
	OP_REQUIRES_OK(context,context->GetAttr("width" ,  &width));
	OP_REQUIRES_OK(context,context->GetAttr("height",  &height));
	OP_REQUIRES_OK(context,context->GetAttr("type"  ,  &type_str));
	OP_REQUIRES_OK(context,context->GetAttr("antialias"  ,  &antialias));
    if( type_str == "NEAREST" )
      type = NEAREST;
    else if (type_str == "LINEAR")
      type = LINEAR;
	else if (type_str == "CUBIC")
	  type = CUBIC;
  }

  void Compute( OpKernelContext* context ) override
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());

	const int num = input_shape.dim_size(0);
	const int channels = input_shape.dim_size(1);
	const int orig_height =  input_shape.dim_size(2);
	const int orig_width =  input_shape.dim_size(3);

	TensorShape out_shape;

	out_shape.AddDim(num);
	out_shape.AddDim(channels);
	out_shape.AddDim(height);
	out_shape.AddDim(width);

    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));

	auto output = output_tensor->flat<T>();
    auto device = context->eigen_gpu_device();

	const float fx = float(orig_width)/float(width);
    const float fy = float(orig_height)/float(height);

	int out_size = width*height*channels*num;
	int out_channelsize = width*height;
	int in_channelsize = orig_width*orig_height;

	if(type==NEAREST){
		NearestNeighborKernel<T><<< LMBOPS_GET_BLOCKS(out_size), LMBOPS_CUDA_NUM_THREADS, 0, device.stream() >>>(
				out_size,
				in_channelsize,
				out_channelsize,
				(T*)input.data(),
				orig_width,
				orig_height,
				fx,
				fy,
				(T*)output.data(),
				width,
				height
				);
		CHECK_CUDA_ERROR;
	}
	else if (type == CUBIC || type == LINEAR)
	{
	  int filter_type;
      if(type == CUBIC)
          filter_type = FILTER_BICUBIC;
      else if(type == LINEAR)
          filter_type = FILTER_TRIANGLE;

	  bool is_downsample = (fx > 1) || (fy > 1);
	  bool should_antialias = antialias && is_downsample;
	  int kernel_width;

      if(filter_type == FILTER_BICUBIC)   kernel_width = 4;
      else if(filter_type == FILTER_BOX)  kernel_width = 1;
      else                                kernel_width = 2;

	  InterpolationKernel<T><<<LMBOPS_GET_BLOCKS(out_size), LMBOPS_CUDA_NUM_THREADS>>>(
			  out_size,
			  in_channelsize,
			  out_channelsize,
			  (T*)input.data(),
			  orig_width,
			  orig_height,
			  fx,
			  fy,
			  (T*)output.data(),
			  width,
			  height,
			  filter_type,
			  kernel_width,
			  should_antialias);
	  CHECK_CUDA_ERROR
	}
	else
	{
	  OP_REQUIRES(context,"Type must be NEAREST, CUBIC or LINEAR", errors::InvalidArgument("Unsupported resampling type") );
	}
  }
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("Resample")                                                         \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ResampleOp_GPU<type>);
REG_KB(float)
REG_KB(double)
#undef REG_KB
