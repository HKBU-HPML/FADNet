#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <iostream>
#include <cmath>
#include <cfloat>


#define ROUND_2_INT(f) ((int)(f >= 0.0 ? (f + 0.5) : (f - 0.5)))


using namespace tensorflow;

REGISTER_OP("FlowOutOfFrame")
	.Input("flow: float32")
	.Input("occ: float32")
	.Output("output: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
      	using namespace ::tensorflow::shape_inference;
		ShapeHandle input_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
		int num      = c->Value(c->Dim(input_shape, 0));
      	int height   = c->Value(c->Dim(input_shape, 2));
      	int width    = c->Value(c->Dim(input_shape, 3));
		c->set_output(0, c->MakeShape({num, 1 , height ,width})) ;
        return Status::OK();
    });


class FlowOutOfFrameOp : public OpKernel {

	public:

  	explicit FlowOutOfFrameOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* c) override {

		TensorShape input_shape = c->input(0).shape();
		int num      = input_shape.dim_size(0);
        int channels = input_shape.dim_size(1);
        int height   = input_shape.dim_size(2);
        int width    = input_shape.dim_size(3);
		int widthHeight = width*height;

        TensorShape output_shape;
        output_shape.AddDim(num);
        output_shape.AddDim(1);
        output_shape.AddDim(height);
        output_shape.AddDim(width);
        Tensor* output_tensor = 0;
        OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output_tensor));


        float *top_ptr = output_tensor->flat<float>().data();
    	const float* bot0_ptr = c->input(0).flat<float>().data();
		const float* bot1_ptr = c->input(1).flat<float>().data();
		for(int n=0; n<num; n++)
		{
			for(int y=0; y<height; y++)
			{
				for(int x=0; x<width; x++)
				{
					float fx = *bot0_ptr;
					float fy = *(bot0_ptr+widthHeight);

					int x2 = ROUND_2_INT(float(x)+fx);
					int y2 = ROUND_2_INT(float(y)+fy);


					if(x2>=0 && y2>=0 && x2<width && y2<height)
					{
						*top_ptr = *bot1_ptr;
					}
					else
					{
						*top_ptr = 1.0;
					}

					if(std::isnan(*bot1_ptr))
						*top_ptr = *bot1_ptr;

					top_ptr++;
					bot1_ptr++;
					bot0_ptr++;
				}
			}
			bot0_ptr+=widthHeight;
		}

	}
};


REGISTER_KERNEL_BUILDER(Name("FlowOutOfFrame").Device(DEVICE_CPU), FlowOutOfFrameOp);


