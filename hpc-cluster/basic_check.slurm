#!/bin/bash
#SBATCH --job-name=ior
#SBATCH --ntasks=4
#SBATCH --output=%x_%j.out

mpirun  hostname
nvidia-smi
