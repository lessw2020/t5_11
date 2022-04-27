# a short code snippet to illustrate how to work with mixed precision on fsdp


# requirements:
# mixed_precision is currently only available in PyTorch nightlies.

import torch

torch.__version__

# expect '1.12.0.dev20220408+cu113' or higher

# import fsdp with Mixed Precision
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

# Mixed precision is it's own dataclass, with three members.
# You create a MixedPrecision instance, and init it with the precision you want to use.
# Below, we are setting all three options to use bfloat16,
# which is preferred where supported over fp16.

bfSixteen = MixedPrecision(
    # parameter precision -
    # note that master params are still fp32...so your final model is still fp32 when saved.
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

# you can use torch.fp16 as well in the above, and mix and match fp32 and bfloat16,etc.

# then simply pass in your mixed_precision policy to the FSDP sharding call:
model = FSDP(
    model,
    auto_wrap_policy=bert_wrap_policy,
    mixed_precision=bfSixteen,
).to(rank)

# that's it!
# Training will use lower precision enabling faster training and in some cases, better results.
