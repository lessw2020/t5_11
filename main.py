# main hq file for t5 training and prediction

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import AutoTokenizer

import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from pathlib import Path

from sklearn.model_selection import train_test_split
import time

# local imports
import verify
import policies

# some globals
g_port = "12369"
g_addr = "localhost"


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch fsdp T5.11 Example")
    parser.add_argument("--save-dir", default="/model_chkpt", type=str)
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 2022)"
    )
    args = parser.parse_args()
    return args


# ----------------   Main functions --------------------
def get_policies(fsdp_unit_params=1000000):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    bf16_ready = verify.bf16_ready

    if bf16_ready:
        mixed_precision_policy = policies.bfSixteen
        print(f"bFloat16 enabled for mixed precision")
    else:
        mixed_precision_policy = policies.fpSixteen
        print(f"bFloat16 support not present. Using fp16 for mixed precision")

    # wrapping policy -------
    wrapping_policy = policies.get_t5_wrapper(fsdp_unit_params)

    return mixed_precision_policy, wrapping_policy


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = g_addr
    os.environ["MASTER_PORT"] = g_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)


def cleanup():
    dist.destroy_process_group()


def setup_tasks(rank, world_size):
    """keep the basic setup list here"""
    setup(rank, world_size)
    # set_printing()
    setup_environ_flags()

    # ---- fsdp main ------------------------------------------------------------


def fsdp_main(rank, world_size, args):
    """main process within each process"""
    setup_tasks(rank, world_size)

    fsdp_unit_params = 1000000

    mp_policy, wrapping_policy = get_policies(fsdp_unit_params)

    model_name = "google/t5-v1_1-base"
    printable_model_name = str.replace(model_name, "/", "==")
    # t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b

    if rank == 0:
        print(f"--> Training for {model_name}")

    dist.barrier()
    cleanup()


# ------------------ Main functions above ------------


if __name__ == "__main__":

    args = parse_args()

    # seed
    torch.manual_seed(args.seed)
    gpus_per_node = torch.cuda.device_count()

    mp.spawn(
        fsdp_main,
        args=(
            gpus_per_node,
            args,
        ),
        nprocs=gpus_per_node,
        join=True,
    )
