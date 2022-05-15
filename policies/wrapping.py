# holds various wrapping policies for fsdp

import torch.distributed as dist
import torch.nn as nn
import torch

from transformers.models.t5.modeling_t5 import T5Block

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import functools
from typing import Type


def transformer_wrapper(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    transformer_layer_cls: Type[nn.Module],
    min_num_params: int = int(1e8),
) -> bool:

    """policy for wrapping transformers with shared embedding
    shared embeddings will be housed in the outermost layer, thus available to all internal
    fsdp units
    """
    is_large = unwrapped_params >= min_num_params
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return is_large and isinstance(module, transformer_layer_cls)


def get_t5_wrapper(min_unit_params=1000000):
    fsdp_wrapping_policy = functools.partial(
        transformer_wrapper,
        min_num_params=min_unit_params,
        transformer_layer_cls=T5Block,
    )
    return fsdp_wrapping_policy
