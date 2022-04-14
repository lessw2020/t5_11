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
import os

# local imports
import verify
import policies

# some globals
g_port = "12369"
g_addr = "localhost"

# setup policies
mp_policy = None
bf16_ready = verify.bf16_ready

if bf16_ready:
    mp_policy = verify.bfSixteen
    print(f"bFloat16 enabled for mixed precision")
else:
    mp_policy = mp_policy.fpSixteen
    print(f"bFloat16 support not present. Using fp16 for mixed precision")


def set_printing():
    """
    disables print when not in rank 0
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if 0== os.getenv("RANK") or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def parse_args():
  parser = argparse.ArgumentParser(description="PyTorch fsdp T5.11 Example")
  parser.add_argument('--save-dir', default = '/model_chkpt',type=str
  parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
  args = parser.parse_args()
  return args

# ----------------   Main functions --------------------

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = g_addr
    os.environ["MASTER_PORT"] = g_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def setup_environ_flags():
  os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)

def cleanup():
    dist.destroy_process_group()


def fsdp_main(rank, world_size, args):
  """ main process within each process """
  setup()
  set_printing()
  setup_environ_flags()

  model_name = "t5"

  print(f"--> Training for {model_name}")






  cleanup()
  pass


# ------------------ Main functions above ------------


if name == __main__:
  
  args = parse_args()
  
  # seed
  torch.manual_seed(args.seed)
  gpus_per_node = torch.cuda.device_count()
 
  mp.spawn(fsdp_main, args=(gpus_per_node, args,), nprocs=gpus_per_node, join=True)
