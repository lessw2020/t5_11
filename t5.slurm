#!/bin/bash

#SBATCH --job-name=T5-trainer

#SBATCH --ntasks=3

#SBATCH --nodes=3

#SBATCH --gpus-per-task=8

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# Enable for A100
#export FI_PROVIDER="efa"


# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="en,eth,em,bond"
dcgmi profile --pause
srun --prolog job_prolog.sh --epilog job_epilog.sh /usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none -x true python -m torch.distributed.run --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 ./main_torchrun.py
dcgmi profile --resume
