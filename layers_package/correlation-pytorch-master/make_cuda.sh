#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda-10.0

cd correlation-pytorch/correlation_package/src
echo "Compiling correlation layer kernels by nvcc..."

# TODO (JEB): Check which arches we need
nvcc -c -o corr_cuda_kernel.cu.o corr_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
nvcc -c -o corr1d_cuda_kernel.cu.o corr1d_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61

cd ../../
python setup.py build install
