from dataclasses import dataclass
from torch.distributed.fsdp import (
    ShardingStrategy,
    BackwardPrefetch,
)

import torch


@dataclass
class train_config:
    # general
    host_port: str = "12368"

    # seed
    seed: int = 2022

    # model
    model_name = "google/t5-v1_1-xxl"  # "google/t5-v1_1-small"
    tokenizer = "t5-large"
    # available models
    ## t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b
    
    model_max_length = 512
    
    #mixed precision
    use_mixed_precision: bool = True
    use_fp16: bool = False
        
    # save models
    save_model: bool = False
    save_folder = "training_checkpoints"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )
    
    # model weights
    model_in_bf16 = False

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    # use rate limiter
    use_rate_limiter: bool = True
    inflight_max = 2
    backward_policy = BackwardPrefetch.BACKWARD_PRE

    # optimizer
    optimizer_type = "childtuning"
    momentum_dtype = torch.float32
    variance_dtype = torch.float32
    use_kahan = False

    # dataloaders
    num_workers_dataloader: int = 0

    # policies
    fsdp_unit_size = 1000000
    use_mixed_precision: bool = True

    # activation checkpointing
    hf_activation_checkpointing: bool = False
    fsdp_activation_checkpointing: bool = True

    # datasets
    dataset_train = "datasets_grammar/gtrain_1k.csv"
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 12
    num_epochs: int = 2

    # validation
    run_validation: bool = False
    val_batch_size = 18
    block_for_validation: bool = False

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True

    # Fine Tuning
    use_child_tuning: bool = True
    use_mirror_optimizer = False
    lr: float = 4e-8

    use_task_free: bool = True
    use_fisher_matrix: bool = False
    percent_F: float = 0.35
