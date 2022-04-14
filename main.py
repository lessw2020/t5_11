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



def parse_args:
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



if name == __main__:
  
  args = parse_args()
  
  # seed
  torch.manual_seed(args.seed)
  gpus_per_node = torch.cuda.device_count()
 
  mp.spawn(fsdp_main, args=(gpus_per_node, args,), nprocs=gpus_per_node, join=True)
