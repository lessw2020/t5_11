from dataclasses import dataclass


@dataclass
class train_config:
    # general
    host_port: str = "12399"

    # model
    model_name = "t5-small"
    save_model: bool = True
    model_checkpoint = "t5small_2e.pt"
    print_sharding_plan: bool = True

    # policies
    fsdp_unit_size = 2000000
    use_mixed_precision: bool = False
    activation_checkpointing: bool = False

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 8
    num_epochs: int = 2

    # logging
    track_memory = True
