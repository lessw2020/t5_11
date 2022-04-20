# main hq file for t5 training and prediction

import os
import argparse
from datasets_grammar.grammar_dataset import grammar
from policies import mixed_precision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


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
)
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import DataLoader
import time

# local imports
import verify
import policies

import datasets_grammar as dg
import tqdm

# config
from defaults import train_config

# some globals
g_port = "12369"
g_addr = "localhost"


def _is_rank_0():
    return 0 == os.getenv("RANK")


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
def get_policies(fsdp_unit_params=1000000):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    bf16_ready = verify.bf16_ready

    if bf16_ready:
        mixed_precision_policy = policies.bfSixteen
        print(f"bFloat16 enabled for mixed precision - using working policy")
    else:
        mixed_precision_policy = policies.fpSixteen
        print(f"bFloat16 support not present. Using fp16 for mixed precision")

    # wrapping policy -------

    wrapping_policy = policies.get_t5_wrapper(fsdp_unit_params)

    return mixed_precision_policy, wrapping_policy


def setup():
    # os.environ["MASTER_ADDR"] = g_addr
    # os.environ["MASTER_PORT"] = cfg.host_port

    # initialize the process group
    dist.init_process_group("nccl")


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache():
    torch.cuda.empty_cache()


def setup_tasks():
    """keep the basic setup list here"""
    setup()
    # clear_gpu_cache() - need to call torch set device first?
    # set_printing()
    setup_environ_flags()


# ----------  Training ----------------------------------------------------------
def train(args, model, local_rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if local_rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)

        """print("************************")
        print(
            "train_loader",
            type(batch),
            batch["source_ids"].size(),
            batch["source_mask"].size(),
            batch["target_ids"].size(),
        )
        print("************************")
        """
        optimizer.zero_grad()
        output = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"],
        )
        # print("##############################")
        # print(output.keys())
        # print("##############################")
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)
        if local_rank == 0:
            inner_pbar.update(1)

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)
    train_accuracy = ddp_loss[0] / ddp_loss[1]
    if local_rank == 0:
        inner_pbar.close()

        print(
            f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
        )  # .format(epoch, train_accuracy))
    return train_accuracy


# ---- Validation ---------------


def test(model, local_rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(local_rank)
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
            pred = output.logits.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(batch["target_ids"].view_as(pred)).sum().item()
            ddp_loss[2] += len(batch)

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                int(ddp_loss[1]),
                int(ddp_loss[2]),
                100.0 * ddp_loss[1] / ddp_loss[2],
            )
        )


# ---- fsdp main ------------------------------------------------------------


def fsdp_main(args):
    """main process within each process"""
    cfg = train_config()  # loads from defaults

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if rank == 0:
        print(f"--> running with these defaults {cfg}")


    fsdp_unit_params = 12000000
    batch_size = 8
    test_batch_size = 4

    mp_policy, wrapping_policy = get_policies(fsdp_unit_params)

    # temp_train()
    # print(f"bailing early...remove")
    # return

    model_name = "t5-small"  # google/t5-v1_1-base"
    printable_model_name = str.replace(model_name, "/", "==")
    # t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b

    # grammar correction
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    train_dataset = dg.get_dataset(tokenizer, None, 512, 512, True)
    if 0 == os.getenv("RANK"):
        print(len(train_dataset))
    # print("bailing")

    # val_dataset = wikihow(tokenizer, "validation", None, 512, 150, True)

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    # sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {"batch_size": batch_size, "sampler": sampler1}
    # test_kwargs = {"batch_size": test_batch_size, "sampler": sampler2}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    # test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    # test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    setup_tasks()


    torch.cuda.set_device(local_rank)
    clear_gpu_cache()

    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    # init_start_event.record()

    # model = model.to(rank)
    # model = DDP(model)
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
    ).to(local_rank)

    if rank == 0:
        print(f"model ")
        fn = printable_model_name + "-shared_layout.txt"
        """with open(fn, "w") as external_file:
            header_text = (
                f"model = {model_name}, sharded with {fsdp_unit_params} parameters\n"
            )
            print(header_text, file=external_file)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            milli_params = total_params * 4 / 1e6
            print(f"\n--> {model_name} has {milli_params} params\n", file=external_file)
            print(f"model wrapping = \n{model}\n\n", file=external_file)

            external_file.close()
        """
    lr = 0.0008
    gamma = 0.7
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    epochs = 1
    best_train_accuracy = float("inf")

    # --- main training loop - todo, this needs to be modularized
    if rank == 0:
        dur = []
        training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        if rank == 0:
            print(f"\n--> Starting Epoch {epoch}")

            t0 = time.time()
        train_accuracy = train(
            args,
            model,
            local_rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler1,
        )
        # test(model, rank, world_size, test_loader)
        scheduler.step()
        if rank == 0:

            dur.append(time.time() - t0)

    # init_end_event.record()
    if rank == 0:
        # inner_pbar.close()
        total_training_time = time.time() - training_start_time
        print(f"Total training time = {total_training_time:.2f}")
        print("Times per epoch:")
        for i, val in enumerate(dur):
            print(f"epoch {i}, time {val:.2f}")
        print()
    # if rank == 0:
        # print(
        # f"Cuda event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        # )
        # print(f"{model}")

        # save block
    save_model = cfg.save_model

    if save_model:
        # dist.barrier()
        states = model.state_dict()
        dist.barrier()

    if rank == 0:
        print(f"--> saving model ...")
        model_save_name = model_name + "train_acc.pt"

        torch.save(states, model_save_name)

        print(f"--> saved {model_save_name} to disk")

    # dist.barrier()
    cleanup()


# ------------------ Main functions above ------------


if __name__ == "__main__":

    args = parse_args()

    # seed
    torch.manual_seed(args.seed)
    gpus_per_node = torch.cuda.device_count()
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
    """

    # mp.spawn(
    #     fsdp_main,
    #     args=(
    #         args,
    #     ),
    #     nprocs=gpus_per_node,
    #     join=True,
    # )
