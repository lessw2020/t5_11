#!/bin/bash

git clone https://github.com/NVIDIA/DCGM
cd DCGM/dcgmbuild || exit
sudo ./build.sh
cd ..
./build.sh -d --rpm
