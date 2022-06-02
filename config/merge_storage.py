from dataclasses import dataclass


@dataclass
class train_config:
    # general
    host_port: str = "12368"

    # model
    model_name = "google/t5-v1_1-base"  # "google/t5-v1_1-small"  #
    save_model: bool = True
    model_checkpoint = "t5small_2cpu.pt"
    print_sharding_plan: bool = True

    # policies
    fsdp_unit_size = 1000000
    use_mixed_precision: bool = True
    activation_checkpointing: bool = True

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 8
    num_epochs: int = 1

    # logging
    track_memory = True
