#!/bin/bash
# build pytorch, follow https://github.com/pytorch/pytorch#from-source

echo "Installing PyTorch dependencies"
conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda113 -y

mkdir -p "${HOME}"/packages
cd "${HOME}"/packages || exit
export USE_SYSTEM_NCCL=1
export TORCH_CUDA_ARCH_LIST="7.0+PTX 8.0"
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch || exit
git checkout release/1.10
git submodule sync
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

python setup.py install

echo "Installing Torchtext"
git clone https://github.com/pytorch/text torchtext
cd torchtext || exit
git checkout release/0.11
git submodule update --init --recursive
python setup.py clean install
