import cuda_layers
from torch.autograd import Function

class Conv2D(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, padding, stride, dilation, groups, hasBias):
        # Save tensors for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.padding = padding
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.hasBias = hasBias

        # Forward pass
        output = cuda_layers.conv2d_forward(input, weight, bias, padding, stride, dilation, groups, hasBias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        input_grad, weight_grad, bias_grad = cuda_layers.conv2d_backward(
            input, weight, grad_output,
            ctx.padding, ctx.stride, ctx.dilation, ctx.groups, ctx.hasBias
        )
        return input_grad, weight_grad, bias_grad, None, None, None, None, None

class Linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, hasBias):
        # save tensors for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.hasBias = hasBias
        # forward
        output = cuda_layers.linear_forward(input, weight, bias, hasBias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        input_grad, weight_grad, bias_grad = cuda_layers.linear_backward(
            input, weight, grad_output, ctx.hasBias
        )
        return input_grad, weight_grad, bias_grad, None