from dataclasses import dataclass


@dataclass
class train_config:
    # general
    host_port: str = "12368"

    # model
    model_name = "t5-large"  # "google/t5-v1_1-small"  #
    save_model: bool = True
    model_checkpoint = "t5_large_save.pt"
    print_sharding_plan: bool = True

    # policies
    fsdp_unit_size = 2000000
    use_mixed_precision: bool = False
    activation_checkpointing: bool = True

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 4
    num_epochs: int = 3

    # validation
    run_validation: bool = True
    val_batch_size = 4

    # logging
    track_memory = True
    memory_report: bool = True
