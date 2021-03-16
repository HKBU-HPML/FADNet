import torch
from torch.autograd import Function
from ..build.lib import GANet
from torch.autograd import Variable
#import GANet
		

class SgaFunction(Function):

    @staticmethod
    def forward(ctx, input, g0, g1, g2, g3):
        ctx.wsize = 5
        ctx.input = input
        ctx.g0 = g0
        ctx.g1 = g1
        ctx.g2 = g2
        ctx.g3 = g3
        assert(input.is_contiguous() == True and g0.is_contiguous() == True and g1.is_contiguous() == True and g2.is_contiguous() == True and g3.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            temp_out = input.new().resize_(num, channels, depth, height, width).zero_()
            mask = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.sga_cuda_forward(input, g0, g1, g2, g3, temp_out, output, mask)
 #           GANet.sga_cuda_forward(input, filters, output, ctx.radius)
            
            output = output.contiguous()
        ctx.save_for_backward(temp_out, mask)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        temp_out, mask = ctx.saved_variables
#        print temp_out.size()
#        print mask.size()
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = ctx.input.size()
            _, _, fsize, _, _ = ctx.g0.size()
#            print fsize            
            gradInput = gradOutput.new().resize_(num, channels, depth, height, width).zero_()
            grad0 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            grad1 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            grad2 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            grad3 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            temp_grad = gradOutput.new().resize_(num, channels, depth, height, width).zero_()     
            max_idx = gradOutput.new().resize_(num, channels, height, width).zero_()    

            GANet.sga_cuda_backward(ctx.input, ctx.g0, ctx.g1, ctx.g2, ctx.g3, temp_out, mask, max_idx, gradOutput, temp_grad, gradInput, grad0, grad1, grad2, grad3)
#            GANet.lga_cuda_backward(ctx.input, ctx.filters, gradOutput, gradInput, gradFilters, ctx.radius)
            gradInput = gradInput.contiguous()
            grad0 = grad0.contiguous()
            grad1 = grad1.contiguous()
            grad2 = grad2.contiguous()
            grad3 = grad3.contiguous()
        return gradInput, grad0, grad1, grad2, grad3
		
		
class Lga3d3Function(Function):

    @staticmethod
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        ctx.input = input
        ctx.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            temp_out1 = input.new().resize_(num, channels, depth, height, width).zero_()
            temp_out2 = input.new().resize_(num, channels, depth, height, width).zero_()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.lga3d_cuda_forward(input, filters, temp_out1, ctx.radius)
            GANet.lga3d_cuda_forward(temp_out1, filters, temp_out2, ctx.radius)
            GANet.lga3d_cuda_forward(temp_out2, filters, output, ctx.radius)
            output = output.contiguous()
        ctx.save_for_backward(temp_out1, temp_out2)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        temp_out1, temp_out2 = ctx.saved_variables
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = ctx.input.size()
            _, _, fsize, _, _ = ctx.filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            GANet.lga3d_cuda_backward(temp_out2, ctx.filters, gradOutput, temp_out2, gradFilters, ctx.radius)
            GANet.lga3d_cuda_backward(temp_out1, ctx.filters, temp_out2, temp_out1, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga3d_cuda_backward(ctx.input, ctx.filters, temp_out1, temp_out2, gradFilters, ctx.radius)
#            temp_out[...] = gradOutput[...]
            temp_out2 = temp_out2.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out2, gradFilters

class Lga3d2Function(Function):

    @staticmethod
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        ctx.input = input
        ctx.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            temp_out = input.new().resize_(num, channels, depth, height, width).zero_()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.lga3d_cuda_forward(input, filters, temp_out, ctx.radius)
            GANet.lga3d_cuda_forward(temp_out, filters, output, ctx.radius)
            output = output.contiguous()
        ctx.save_for_backward(temp_out)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        temp_out, = ctx.saved_variables
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = ctx.input.size()
            _, _, fsize, _, _ = ctx.filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            GANet.lga3d_cuda_backward(temp_out, ctx.filters, gradOutput, temp_out, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga3d_cuda_backward(ctx.input, ctx.filters, temp_out, gradOutput, gradFilters, ctx.radius)
            temp_out[...] = gradOutput[...]
            temp_out = temp_out.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out, gradFilters

class Lga3dFunction(Function):
       
    @staticmethod
    def forward(ctx, input, filters, radius=2):
        ctx.radius = radius
        ctx.input = input
        ctx.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.lga3d_cuda_forward(input, filters, output, ctx.radius)
            output = output.contiguous()
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = ctx.input.size()
            _, _, fsize, _, _ = ctx.filters.size()
            gradInput = gradOutput.new().resize_(num, channels, depth, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            GANet.lga3d_cuda_backward(ctx.input, ctx.filters, gradOutput, gradInput, gradFilters, ctx.radius)
            gradInput = gradInput.contiguous()
            gradFilters = gradFilters.contiguous()
        return gradInput, gradFilters

class Lga3Function(Function):

    @staticmethod
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        ctx.input = input
        ctx.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            temp_out1 = input.new().resize_(num, channels, height, width).zero_()
            temp_out2 = input.new().resize_(num, channels, height, width).zero_()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, temp_out1, ctx.radius)
            GANet.lga_cuda_forward(temp_out1, filters, temp_out2, ctx.radius)
            GANet.lga_cuda_forward(temp_out2, filters, output, ctx.radius)
            output = output.contiguous()
        ctx.save_for_backward(temp_out1, temp_out2)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        temp_out1, temp_out2 = ctx.saved_variables
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = ctx.input.size()
            _, fsize, _, _ = ctx.filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(temp_out2, ctx.filters, gradOutput, temp_out2, gradFilters, ctx.radius)
            GANet.lga_cuda_backward(temp_out1, ctx.filters, temp_out2, temp_out1, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga_cuda_backward(ctx.input, ctx.filters, temp_out1, temp_out2, gradFilters, ctx.radius)
#            temp_out[...] = gradOutput[...]
            temp_out2 = temp_out2.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out2, gradFilters

class Lga2Function(Function):

    @staticmethod
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        ctx.input = input
        ctx.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            temp_out = input.new().resize_(num, channels, height, width).zero_()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, temp_out, ctx.radius)
            GANet.lga_cuda_forward(temp_out, filters, output, ctx.radius)
            output = output.contiguous()
        ctx.save_for_backward(temp_out)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        temp_out, = ctx.saved_variables
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = ctx.input.size()
            _, fsize, _, _ = ctx.filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(temp_out, ctx.filters, gradOutput, temp_out, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga_cuda_backward(ctx.input, ctx.filters, temp_out, gradOutput, gradFilters, ctx.radius)
            temp_out[...] = gradOutput[...]
            temp_out = temp_out.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out, gradFilters, None

class LgaFunction(Function):
       
    @staticmethod
    def forward(ctx, input, filters, radius=2):
        ctx.radius = radius
        ctx.input = input
        ctx.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, output, ctx.radius)
            output = output.contiguous()
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = ctx.input.size()
            _, fsize, _, _ = ctx.filters.size()
            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(ctx.input, ctx.filters, gradOutput, gradInput, gradFilters, ctx.radius)
            gradInput = gradInput.contiguous()
            gradFilters = gradFilters.contiguous()
        return gradInput, gradFilters

class MyLoss2Function(Function):

    @staticmethod
    def forward(ctx, input1, input2, thresh=1, alpha=2):
        ctx.thresh = thresh
        ctx.alpha = alpha
        ctx.diff = input1 - input2
        temp=torch.abs(ctx.diff)
        temp[temp < ctx.thresh] = temp[temp < ctx.thresh] ** 2 / ctx.thresh
        tag = (temp <= ctx.thresh + ctx.alpha) & (temp >= ctx.thresh)
        temp[tag]=temp[tag] * 2 - (temp[tag] - ctx.thresh) ** 2 /(2.0 * ctx.alpha) - ctx.thresh
        temp[temp > ctx.thresh + ctx.alpha] += (ctx.alpha / 2.0)
        
        return torch.mean(temp)

    @staticmethod
    def backward(ctx, gradOutput):
        scale = torch.abs(ctx.diff)
        scale[scale > ctx.thresh + ctx.alpha] = 1
        tag = (scale <= ctx.thresh+ctx.alpha) & (scale >= ctx.thresh)
        scale[tag] = 2 - (scale[tag] - ctx.thresh) / ctx.alpha
        tag = scale < ctx.thresh
        scale[tag] = 2*scale[tag] / ctx.thresh
        ctx.diff[ctx.diff > 0] = 1.0
        ctx.diff[ctx.diff < 0] = -1.0
        ctx.diff = ctx.diff * scale * gradOutput / scale.numel()
        return ctx.diff, Variable(torch.Tensor([0]))

class MyLossFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, upper_thresh=5, lower_thresh=1):
        ctx.upper_thresh = upper_thresh
        ctx.lower_thresh = lower_thresh
        ctx.diff = input1 - input2
        return torch.mean(torch.abs(ctx.diff))

    @staticmethod
    def backward(ctx, gradOutput):
        scale = torch.abs(ctx.diff)
        scale[scale > ctx.upper_thresh] = 1
        tag = (scale <= ctx.upper_thresh) & (scale >= ctx.lower_thresh)
        scale[tag] = 2 - torch.abs(scale[tag]-(ctx.upper_thresh + ctx.lower_thresh)/2.)/2.
        ctx.diff[ctx.diff > 0] = 1
        ctx.diff[ctx.diff < 0] = -1
        ctx.diff = ctx.diff * scale * gradOutput
        return ctx.diff, Variable(torch.Tensor([0]))

