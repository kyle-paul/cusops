# Custom Operations for Deep Learning

## Introduction
In this repository, I attempt to reimplement popular operations such as convolution, linear layer, etc from scratch with CUDA programming (for GPU) and C++ (for CPU) to enhance my deep knowledge about Deep Learning. Then, I'll try to modify these operations and implement novel custom operations aiming at reducing inference time and increasing efficiency:

### Convolution 2D
```
\begin{gather}
	\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
	\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
\end{gather}
```

## Setup environment
In order to run the build the package `cuda_layers` or `cpp_layers` code in `setup.py` file, you need to use Docker or your own environment. For those want to use docker:

```bash
docker pull kylepaul/deeplearning:deployment
```

More information of this docker image is at [here](`https://hub.docker.com/repository/docker/kylepaul/deeplearning/tags`). Then you would want to follow my `compose.yml` file for intitializing docker container and runnning with `docker compose up -d`

## Build custom operations packages
The source code of convolution and linear layers are implemented in the `src` folder, in which you can find both `cpp` version runninng on CPU or `cuda` version running on GPU utilizing parallel computation with kernels implementation. The `cuda_layers` contain the code of backward that can be registered into the mechanism `torch.autograd` of Pytorch. To install the package, go into the `cuda_layers`:

```bash
python setup.py install
```

## Train on MNIST
All the trainining code and custom operation registration, both forward and backward pass, was in the file `modules.py`. I use the default mnist dataset from `torchvision` to test the operations. To run the code train, simply:

```bash
python train.py
```