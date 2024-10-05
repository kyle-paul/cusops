#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace F = torch::nn::functional;
namespace py = pybind11;

torch::Tensor conv2d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    bool hasBias, int padding, int stride, int dilation, int groups
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
    torch::Tensor padded_input = F::pad(input, F::PadFuncOptions({padding, padding, padding, padding}));
    torch::Tensor output = torch::empty({batch_size, out_channels, new_height, new_width});

    float accumulation, A, B;
    for (int b=0; b < batch_size; b++) {
        for (int oc=0; oc < out_channels; oc++) {
            for (int h=0; h < new_height; h++) {
                for (int w=0; w < new_width; w++) {
                    accumulation = 0;
                    for (int ic=0; ic < in_channels; ic++) {
                        for (int k1=0; k1 < kernel_size; k1++) {
                            for (int k2=0; k2 < kernel_size; k2++) {
                                A = padded_input.index({b, ic, h+k1, w+k2}).item<float>();
                                B = weight.index({oc, ic, k1, k2}).item<float>();
                                accumulation += (A * B);
                            }
                        }
                    }
                    if (hasBias) accumulation += bias.index({b}).item<float>();
                    output.index({b, oc, h, w}) = accumulation;
                }
            }
        }
    }
    return output;
}

torch::Tensor linear_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    bool hasBias
)
{
    int batch_size = input.size(0);
    int dim = input.size(1);
    int new_dim = weight.size(0);
    torch::Tensor transposed_weight = torch::transpose(weight, 0, 1);
    torch::Tensor output = torch::empty({batch_size, new_dim});

    float accumulation, A, B;
    for (int b=0; b < batch_size; b++) {
        for (int d2=0; d2 < new_dim; d2++) {
            accumulation = 0;
            for (int d1=0; d1 < dim; d1++) {
                A = input.index({b, d1}).item<float>();
                B = transposed_weight.index({d1, d2}).item<float>();
                accumulation += (A * B);
            }
            if (hasBias) accumulation += bias.index({d2}).item<float>();
            output.index({b,d2}) = accumulation;
        }
    }
    return output;
}

PYBIND11_MODULE(cpp_layers, m) {
    m.doc() = "Implementation of custom operation in C++";
    m.def("conv2d_forward", &conv2d_forward, "conv2d_forward");
    m.def("linear_forward", &linear_forward, "linear_forward");
}