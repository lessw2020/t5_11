# main hq file for t5 training and prediction

import os
import argparse

from platformdirs import user_config_dir
from psutil import net_if_addrs
from datasets_grammar.grammar_dataset import grammar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from statistics import stdev

# from torchvision import datasets, transforms


# for grammar correction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# for generation
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq

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
    FullStateDictConfig,
    StateDictType,
)


from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from policies import mixed_precision

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import DataLoader

# from nlp import load_metric
# from nlp import load_dataset
from ChildTuningOptimizer import ChildTuningAdamW

# from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# local imports
import verify
import policies

import datasets_grammar as dg
import tqdm

# config
import config
from utils.calculations_utils import calc_flop

import performance

# some globals
g_port = "12369"
g_addr = "localhost"


def _is_rank_0():
    return 0 == os.getenv("RANK")


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


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
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    args = parser.parse_args()
    return args


# ----------------   Main functions --------------------
def get_policies(cfg, rank):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.use_mixed_precision:
        bf16_ready = verify.bf16_ready

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if rank == 0:
                print(f"FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    # wrapping policy -------
    # print(f"**overriding mp to fp16 - remove")
    # mixed_precision_policy = policies.fpSixteen

    wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy, wrapping_policy


def setup(rank, world_size, cfg):
    # os.environ["MASTER_ADDR"] = g_addr
    # os.environ["MASTER_PORT"] = cfg.host_port

    # initialize the process group
    dist.init_process_group("nccl")  # , rank=rank, world_size=world_size)


def setup_environ_flags(cfg, rank):

    if rank == 0:
        print("\n--> Setting environment policies....\n")
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if cfg.distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if rank == 0:
            print(f"--> running with torch dist debug set to detail, rank {rank}")
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["PYTHONFAULTHANDLER"] = str(1)


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f"clearing cache for rank {rank}")
    torch.cuda.empty_cache()


def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup(rank, world_size, cfg)
    # clear_gpu_cache() - need to call torch set device first?
    # set_printing()
    setup_environ_flags(cfg, rank)


def format_stats(item, rounding=8):
    return round(item, ndigits=rounding)


# ----------  Training ----------------------------------------------------------
def train(
    cfg,
    model,
    local_rank,
    rank,
    world_size,
    train_loader,
    optimizer,
    epoch,
    sampler=None,
    profiler=None,
    scaler=None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)

        optimizer.zero_grad()
        output = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"],
        )

        loss = output["loss"]
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # adjust scaling for next minibatch
        else:
            loss.backward()
            optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)
        if rank == 0:
            inner_pbar.update(1)
        if profiler:
            profiler.step()

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    train_accuracy = ddp_loss[0] / ddp_loss[1]
    if rank == 0:
        inner_pbar.close()

        print(
            f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
        )  # .format(epoch, train_accuracy))
    return train_accuracy


# ---- Validation ---------------


def validation(cfg, model, local_rank, rank, world_size, test_loader, scaler):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(test_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in test_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"],
            )
            ddp_loss[0] += output["loss"].item()  # sum up batch loss
            ddp_loss[1] += len(batch)

            if rank == 0:
                inner_pbar.update(1)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    val_loss = ddp_loss[0] / ddp_loss[1]

    if rank == 0:
        # test_loss = ddp_loss[0] / ddp_loss[1]
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def sync_all_device():
    # setup() has already configured CUDA_VISIBLE_DEVICES such that each
    # process exclusively works on its own set of devices. So it's safe to
    # do device sync here
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


# ---- fsdp main ------------------------------------------------------------


def fsdp_main(args):
    """main process within each process"""
    cfg = config.benchmark_config()  # loads from defaults

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> Running with benchmark configs!")
        print(f"--> running with these defaults {cfg}")
        time_of_run = get_date_of_run()

    setup_tasks(rank, world_size, cfg)

    if rank == 0:
        print(f"settings for NCCL = {os.getenv('NCCL_DEBUG')}")

    fsdp_unit_params = cfg.fsdp_unit_size

    num_wrapping = policies.get_size_policy()  # (fsdp_unit_params)

    batch_size = cfg.batch_size
    if rank == 0:
        print(f"\n BatchSize = {batch_size}\n")

    val_batch_size = cfg.val_batch_size

    scaler = None  # only used for fp16

    mp_policy, wrapping_policy = get_policies(cfg, rank)

    if cfg.use_fp16:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        scaler = ShardedGradScaler()

    model_name = cfg.model_name  # "google/t5-v1_1-small"  #   #
    if rank == 0:
        print(f"--> training for model {model_name}")

    printable_model_name = str.replace(model_name, "/", "=")
    save_name = model_name + "-"

    # t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b

    # grammar correction
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer, model_max_length=cfg.model_max_length
    )

    # setup gpu memory monitoring
    if rank == 0:
        memmax = performance.Memory_Maximizer()

    if cfg.hf_activation_checkpointing:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_cache=False)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # summarization
    # model = T5ForConditionalGeneration.from_pretrained(model_name)
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # dataset_name = "jfleg_train.csv"

    if rank == 0:
        print(f"--> Training for {model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {model_name} has {total_params/1e6} Million params\n")

        # print(f"{dataset_name} contains: {dataset.keys()}")
        # print("Size of {dataset_name} train dataset: ", dataset["train"].shape)
        # print(
        #    "Size of {dataset_name} Validation dataset: ", dataset["validation"].shape
        # )

    # ____________ create batch dataset
    train_name = None
    if cfg.dataset_train:
        train_name = cfg.dataset_train

    train_dataset = dg.get_dataset(tokenizer, train_name, 512, 512, True)
    if 0 == os.getenv("RANK"):
        print(f"--> Training Set Len = {len(train_dataset)}")
        print(f"using dataset {train_name}")
    # print("bailing")

    val_dataset = dg.get_dataset(tokenizer, cfg.dataset_test, 512, 512, True)
    if 0 == os.getenv("RANK"):
        print(f"--> Validation set len = {len(val_dataset)}")
        print(f"using dataset {cfg.dataset_test}")

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    if rank == 0:
        print(f"batch size = {batch_size}")

    train_kwargs = {"batch_size": batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": val_batch_size, "sampler": sampler2}
    cuda_kwargs = {
        "num_workers": cfg.num_workers_dataloader,
        "pin_memory": False,
        "shuffle": False,
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    # init_start_event.record()

    # model = model.to(rank)
    # model = DDP(model)
    if cfg.hf_activation_checkpointing:
        model.gradient_checkpointing_enable()
        print(f"HF Activation checkpointing enabled\n")

    model_config = model.config
    embedding_size = (model.state_dict()["shared.weight"].shape)[1]
    FLOP = calc_flop(cfg, model_config, cfg.model_max_length, embedding_size)

    print(
        "embedding size **********",
        model.state_dict()["shared.weight"].shape,
        embedding_size,
    )
    if cfg.pure_bfloat:
        model = model.to(torch.bfloat16)
        mp_policy = None
        if rank == 0:
            print(f"--> Model converted to BF16.\nRunning in PURE BFloat mode")

    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
    )
    # print(f"IMPORTANT - No mixed precision policy -fp32 running")
    # move model to gpu
    # model.to(local_rank)

    # sanity check
    if cfg.hf_activation_checkpointing and cfg.fsdp_activation_checkpointing:
        print(
            f"alert! --> fsdp and hf chekckpointing both active...this is not compatible. Aborting."
        )
        return

    if cfg.fsdp_activation_checkpointing:
        policies.apply_fsdp_checkpointing(model)

    if rank == 0 and cfg.print_sharding_plan:
        print(f"model ")
        fn = printable_model_name + "-sharded_layout.txt"
        with open(fn, "w") as external_file:
            header_text = (
                f"model = {model_name}, sharded with {fsdp_unit_params} parameters\n"
            )
            print(header_text, file=external_file)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            milli_params = total_params * 4 / 1e6
            print(
                f"\n--> {model_name} has {milli_params} Million params\n",
                file=external_file,
            )
            print(f"model wrapping = \n{model}\n\n", file=external_file)

            external_file.close()

    lr = 0.0008
    gamma = 0.85
    if cfg.use_task_free:
        optimizer = ChildTuningAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            reserve_p=cfg.percent_F,
            mode="taskfree",
        )
    elif cfg.pure_bfloat:
        from bff_optimizer import BFF_Optimizer

        optimizer = BFF_Optimizer(
            model.parameters(), lr=lr, weight_decay=0.0, pure_mode=True
        )
        if rank == 0:
            print("Optimizer set to BFF Optimizer")
    elif cfg.pure_optimizer:
        from bff_optimizer import BFF_Optimizer

        optimizer = BFF_Optimizer(
            model.parameters(), lr=lr, weight_decay=0.0, pure_mode=False
        )
        if rank == 0:
            print("Optimizer set to BFF Optimizer with only pure OPTIMIZER STATES")

    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        if rank == 0:
            print(f"--> optimizer is AdamW")

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    epochs = cfg.num_epochs
    if rank == 0:
        print(f"Training for {epochs} epochs")

    best_train_accuracy = float("-inf")
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    # --- main training loop - todo, this needs to be modularized

    if rank == 0:
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        memmax.start()

        training_start_time = time.time()

    torch_profiler = None
    """with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "fsdp_v100/profile_traces"
        ),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    ) as torch_profiler:
    """
    if rank == 0 and cfg.track_memory:
        fn = cfg.model_name + "memory_tracking.txt"
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    start_training_time = time.time()
    for epoch in range(1, epochs + 1):
        if rank == 0:
            print(f"\n--> Starting Epoch {epoch}")

            t0 = time.time()
        train_accuracy = train(
            cfg,
            model,
            local_rank,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler1,
            profiler=torch_profiler,
            scaler=scaler,
        )
        if rank == 0:
            memmax.update()

        if cfg.block_for_validation:
            dist.barrier()
            if rank == 0:
                print(f"--> blocking ranks for pre-validation synching...")

        if cfg.run_validation:
            curr_val_loss = validation(
                cfg, model, local_rank, rank, world_size, test_loader, scaler=scaler
            )

        scheduler.step()

        if rank == 0:
            print(f"--> epoch {epoch} completed...entering save and stats zone")

            total_epoch_time = time.time() - t0
            print(f"epoch_time = {total_epoch_time}")
            dur.append(total_epoch_time)

            train_acc_tracking.append(format_stats(train_accuracy.item()))
            net_flops = format_stats(FLOP / 10**12 / total_epoch_time, rounding=6)
            print(f"TFLOP/s/GPU: {net_flops}")

            if cfg.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if cfg.track_memory:
                mem_alloc_tracker.append(torch.cuda.memory_allocated())
                mem_reserved_tracker.append(torch.cuda.memory_reserved())

        """if cfg.save_model and curr_val_loss < best_val_loss:
            # update curr best val accuracy

            # save
            # todo -remove
            dist.barrier()
            if rank == 0:
                print(f"--> dist barrier activated")
            if rank == 0:
                print(f"--> entering save model state...")
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            # states = model.state_dict()
            print(f"saving process: rank {rank}  done w state_dict")
            # dist.barrier()
            # print(f"rank {rank}  done w 2nd barrier")

            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                model_save_name = "-" + time_of_run + "-" + save_name + currEpoch

                torch.save(cpu_state, model_save_name)

                print(f"--> saved {model_save_name} to disk")
            # todo- remove
            dist.barrier()
            if rank == 0:
                print(f"--> dist barrier removed.")
        """
        # announce new val loss record:
        if rank == 0 and curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            print(f"-->>>> New Val Loss Record: {best_val_loss}")

    sync_all_device()
    end_training_time = time.time()

    delays = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(
        delays, (end_training_time - start_training_time) / epochs
    )
    for i, item in enumerate(delays):
        delays[i] = round(item, 4)

    if rank == 0:
        print("Flops cnt and delays", FLOP, delays)

        gflops_gpu = FLOP / 10**9 * np.reciprocal(np.array(delays))
        tflops_gpu = FLOP / 10**12 * np.reciprocal(np.array(delays))
        print(f"gflops per gpu={gflops_gpu}")

    # init_end_event.record()
    if rank == 0:
        # inner_pbar.close()
        total_training_time = time.time() - training_start_time
        print(f"Total training time = {total_training_time:.2f}")
        print("Times per epoch:")
        for i, val in enumerate(dur):
            print(f"epoch {i}, time {val:.2f}")
        print()

        # memory
        memmax.stop()

        if cfg.track_memory:
            print(f"total memory reserved: {mem_reserved_tracker}")
            print(f"total memory allocated: {mem_alloc_tracker}")

        print(f"Training accuracy: {train_acc_tracking}")
        if cfg.run_validation:
            print(f"Validation accuracy: {val_acc_tracking}")
            print(f"\n Best Val accuracy: {best_val_loss}")

        # memory summary
        if cfg.memory_report and rank == 0:
            print(
                f"CUDA Memory Summary After Last training:\n {torch.cuda.memory_summary()}"
            )
        if rank == 0:

            # print("LEN Tflops",len(tflops_gpu), sum(tflops_gpu), tflops_gpu)
            print(
                f"gflops/gpu = {sum(gflops_gpu) / len(gflops_gpu):.2f} ({stdev(gflops_gpu):.2f})\n"
                f"Tflops/gpu = {sum(tflops_gpu) / len(tflops_gpu):.2f} ({stdev(tflops_gpu):.2f})\n"
            )
        # print(
        # f"Cuda event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        # )
        # print(f"{model}")

        # save block
        # save_model = cfg.save_model

        # debug hang
        # runs on all ranks
        # print(f"rank {rank} calling barrier")
        # dist.barrier()
        # print(f"rank {rank} done w barrier, calling state_dict")

    dist.barrier()
    cleanup()


# ------------------ Main functions above ------------


if __name__ == "__main__":

    args = parse_args()

    # seed
    torch.manual_seed(2022)
    gpus_per_node = torch.cuda.device_count()

    # torch run start
    fsdp_main(args)

    # cache workaround
    """ dataset_name = "grammar_train.csv"
    full_path_dataset = Path.cwd()/'datasets_grammar'/dataset_name

    temp_full_dataset = load_dataset(
        "csv",
        data_files={
            "train": [full_path_dataset]
        },  # "eval": "grammar_validation.csv"},
        delimiter=",",
    )
    print(f"temp dset loaded in main = len {len(temp_full_dataset)}")
    

    mp.spawn(
        fsdp_main,
        args=(
            gpus_per_node,
            args,
        ),
        nprocs=gpus_per_node,
        join=True,
    )
    """
