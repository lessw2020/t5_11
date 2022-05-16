from dataclasses import dataclass


@dataclass
class train_config:
    # general
    host_port: str = "12368"

    # model
    model_name = "google/t5-v1_1-xl"  # "google/t5-v1_1-small"
    tokenizer = "t5-large"
    # available models
    ## t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b
    save_model: bool = True
    model_checkpoint = "t5_small_save.pt"
    print_sharding_plan: bool = False

    # dataloaders
    num_workers_dataloader: int = 2

    # policies
    fsdp_unit_size = 1000000
    use_mixed_precision: bool = True

    activation_checkpointing: bool = True

    # datasets
    dataset_train = "datasets_grammar/gtrain_150K.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 8
    num_epochs: int = 14

    # validation
    run_validation: bool = True
    val_batch_size = 4
    block_for_validation: bool = False

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True

    # Fine Tuning
    use_child_tuning: bool = True

    use_task_free: bool = False
    use_fisher_matrix: bool = True
    percent_F: float = 0.75
