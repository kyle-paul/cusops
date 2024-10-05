#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace py = pybind11;

__global__ void conv2d_forward_kernel(
    float* input, float* weight, float* bias, float* output,
    int in_channels, int out_channels, 
    int height, int width, int new_height, int new_width,
    int kernel_size, int padding, int stride, int dilation, int groups, bool hasBias
)
{
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;

    float accumulation = 0.0f;
    float A, B;
    for (int ic = 0; ic < in_channels; ic++) { 
        for (int k1 = 0; k1 < kernel_size; k1++) {
            for (int k2 = 0; k2 < kernel_size; k2++) {
                int in_h = h * stride + k1 - padding;
                int in_w = w * stride + k2 - padding;
                if (in_h >= 0 && in_w >= 0 && in_h < height && in_w < width) {
                    A = input[(b * in_channels + ic) * height * width + in_h * width + in_w];
                    B = weight[(oc * in_channels + ic) * kernel_size * kernel_size + k1 * kernel_size + k2];
                    accumulation += (A * B);
                }
            }
        }
    }
    if (hasBias && bias != nullptr) {
        accumulation += bias[oc];
    }
    output[(b * out_channels + oc) * new_height * new_width + h * new_width + w] = accumulation;
}

torch::Tensor conv2d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int padding, int stride, int dilation, int groups, bool hasBias
)
{
    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int in_channels = weight.size(1);
    int kernel_size = weight.size(2);

    int new_height = ((height - kernel_size + 2 * padding) / stride) + 1;
    int new_width = ((width - kernel_size + 2 * padding) / stride) + 1;
    torch::Tensor output = torch::zeros({batch_size, out_channels, new_height, new_width}).to(torch::kCUDA);

    dim3 blockDim(new_height, new_width);
    dim3 gridDim(batch_size, out_channels);

    conv2d_forward_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        hasBias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        in_channels, out_channels,
        height, width, new_height, new_width,
        kernel_size, padding, stride, dilation, groups, hasBias
    );
    return output;
}

__global__ void linear_forward_kernel(
    float* input, float* weight, float* output, float* bias, 
    int dim, int new_dim, bool hasBias
)
{
    int b = blockIdx.x;
    int d2 = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (d2 < new_dim) {
        float accumulation = 0.0f;
        float A, B;
        for (int d1=0; d1 < dim; d1++) {
            A = input[b * dim + d1];
            B = weight[d1 * new_dim + d2];
            accumulation += (A * B);
        }
        if (hasBias && bias != nullptr) {
            accumulation += bias[d2];
        }  
        output[b * new_dim + d2] = accumulation;
    }
}


torch::Tensor linear_forward(
    torch::Tensor input, torch::Tensor weight, 
    torch::Tensor bias, bool hasBias
)
{
    int batch_size = input.size(0);
    int dim = input.size(1);
    int new_dim = weight.size(0);
    torch::Tensor transposed_weight = torch::transpose(weight, 0, 1);
    torch::Tensor output = torch::empty({batch_size, new_dim}).to(torch::kCUDA);

    int threads = 1024;
    int blocks_y = (new_dim + threads - 1) / threads;

    dim3 blockDim(threads);
    dim3 gridDim(batch_size, blocks_y);

    linear_forward_kernel<<<gridDim, blockDim>>> (
        input.data_ptr<float>(),
        transposed_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        hasBias ? bias.data_ptr<float>() : nullptr,
        dim, new_dim, hasBias
    );
    return output;
}


__global__ void conv2d_backward_kernel (
    float *input, float *weight, 
    float *output_grad, float *input_grad, float *weight_grad, float *bias_grad,
    int in_channels, int out_channels, 
    int height, int width, int new_height, int new_width,
    int kernel_size, int padding, int stride, int dilation, int groups, bool hasBias
)
{
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;

    // compute the gradient for input
    for (int ic=0; ic < in_channels; ic++) {
        for (int k1=0; k1 < kernel_size; k1++){
            for (int k2=0; k2 < kernel_size; k2++) {
                int in_h = h * stride + k1 - padding;
                int in_w = w * stride + k2 - padding;
                if (in_h >= 0 && in_w >= 0 && in_h < height && in_w < width) {
                    input_grad[(b * in_channels + ic) * height * width + in_h * width + in_w] += 
                        output_grad[(b * out_channels + oc) * new_height * new_width + h * new_width + w] *
                        weight[(oc * in_channels + ic) * kernel_size * kernel_size + k1 * kernel_size + k2];
                }
            }
        }
    }

    // compute the gradient for weight
    for (int ic=0; ic < in_channels; ic++) {
        for (int k1 = 0; k1 < kernel_size; k1++) {
            for (int k2 = 0; k2 < kernel_size; k2++) {
                int in_h = h * stride + k1 - padding;
                int in_w = w * stride + k2 - padding;
                if (in_h >= 0 && in_w >= 0 && in_h < height && in_w < width) {
                    weight_grad[(oc * in_channels + ic) * kernel_size * kernel_size + k1 * kernel_size + k2] +=
                        output_grad[(b * out_channels + oc) * new_height * new_width + h * new_width + w] *
                        input[(b * in_channels + ic) * height * width + in_h * width + in_w];
                }
            }
        }
    }

    // Compute gradient for bias
    if (hasBias) {
        atomicAdd(&bias_grad[oc], output_grad[(b * out_channels + oc) * new_height * new_width + h * new_width + w]);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv2d_backward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor output_grad,
    int padding, int stride, int dilation, int groups, bool hasBias
) 
{
    int batch_size = input.size(0);
    int in_channels = weight.size(1);
    int out_channels = weight.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int new_height = output_grad.size(2);
    int new_width = output_grad.size(3);
    int kernel_size = weight.size(2);

    torch::Tensor input_grad = torch::empty({input.size(0), input.size(1), input.size(2), input.size(3)}).to(torch::kCUDA);
    torch::Tensor weight_grad = torch::empty({weight.size(0), weight.size(1), weight.size(2), weight.size(3)}).to(torch::kCUDA);
    torch::Tensor bias_grad = (hasBias) ? torch::empty({weight.size(0)}).to(torch::kCUDA) : torch::Tensor();
    dim3 blockDim (new_height, new_width);
    dim3 gridDim (batch_size, out_channels);

    conv2d_backward_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output_grad.data_ptr<float>(),
        input_grad.data_ptr<float>(),
        weight_grad.data_ptr<float>(),
        hasBias ? bias_grad.data_ptr<float>() : nullptr,
        in_channels, out_channels,
        height, width, new_height, new_width,
        kernel_size, padding, stride, dilation, groups, hasBias
    );
    
    if (hasBias)
        return std::make_tuple(input_grad, weight_grad, bias_grad);
    else
        return std::make_tuple(input_grad, weight_grad, torch::Tensor());
}


__global__ void linear_backward_kernel(
    float *input, float *weight, 
    float* output_grad, float *input_grad, float* weight_grad, float *bias_grad,
    int dim, int new_dim, bool hasBias
)
{
    int b = blockIdx.x;
    int d2 = blockIdx.y * blockDim.x + threadIdx.x;

    if (d2 < new_dim) {
        float accumulation = 0.0f;
        for (int d1=0; d1 < dim; d1++) {
            input_grad[b * dim + d1] += weight[d1 * new_dim + d2] * output_grad[b * new_dim + d2];
        }
        for (int d1=0; d1 < dim; d1++) {
            weight_grad[d1 * new_dim + d2] += input[b * dim + d1] * output_grad[b * new_dim + d2];
        }
        if (hasBias && bias_grad != nullptr) {
            atomicAdd(&bias_grad[d2], output_grad[b * new_dim + d2]);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_backward(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor output_grad, bool hasBias
) 
{
    torch::Tensor input_grad = torch::empty({input.size(0), input.size(1)}).to(torch::kCUDA);
    torch::Tensor transposed_weight = torch::transpose(weight, 0, 1);
    torch::Tensor weight_grad = torch::empty({transposed_weight.size(0), transposed_weight.size(1)}).to(torch::kCUDA);
    torch::Tensor bias_grad = (hasBias) ? torch::empty({transposed_weight.size(1)}).to(torch::kCUDA) : torch::Tensor();
    int batch_size = input.size(0);
    int dim = input.size(1);
    int new_dim = weight.size(0);

    int threads = 1024;
    int blocks_y = (new_dim + threads - 1) / threads;
    
    dim3 blockDim(threads);
    dim3 gridDim(batch_size, blocks_y);

    linear_backward_kernel<<<gridDim, blockDim>>> (
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output_grad.data_ptr<float>(),
        input_grad.data_ptr<float>(),
        weight_grad.data_ptr<float>(),
        hasBias ? bias_grad.data_ptr<float>() : nullptr,
        dim, new_dim, hasBias
    );

    if (hasBias)
        return std::make_tuple(input_grad, weight_grad, bias_grad);
    else
        return std::make_tuple(input_grad, torch::transpose(weight_grad, 0, 1), torch::Tensor());
}
    


PYBIND11_MODULE(cuda_layers, m) {
    m.doc() = "Implementation of custom operation in C++";
    m.def("conv2d_forward", &conv2d_forward, "conv2d_forward (CUDA)");
    m.def("linear_forward", &linear_forward, "linear forward (CUDA)");
    m.def("conv2d_backward", &conv2d_backward, "conv2d_backward (CUDA)");
    m.def("linear_backward", &linear_backward, "linear backward (CUDA)");
}