#!/bin/bash
# build pytorch, follow https://github.com/pytorch/pytorch#from-source

export CUDA_HOME=/usr/local/cuda
export NCCL_HOME=/home/ec2-user/packages/nccl/build
export MPI_HOME=/opt/amazon/openmpi
export LD_LIBRARY_PATH=${NCCL_HOME}/lib:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${MPI_HOME}/lib:${MPI_HOME}/lib64:/usr/local/lib:/usr/lib
export PATH=${NCCL_HOME}/bin:${CUDA_HOME}/bin:${PATH}
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc

echo "Installing PyTorch dependencies"
conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda113 -y

cd /lustre || exit
export USE_SYSTEM_NCCL=1
export TORCH_CUDA_ARCH_LIST="7.0+PTX 8.0"
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch || exit
git checkout master
git submodule sync
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

python setup.py install
