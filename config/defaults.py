from dataclasses import dataclass


@dataclass
class train_config:
    # general
    host_port: str = "12399"

    # model
    model_name = "t5-small"
    save_model: bool = True
    model_checkpoint = "t5large_2e.pt"

    # policies
    fsdp_unit_size = 2000000
    mixed_precision: bool = True
    activation_checkpointing: bool = True

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 16
    num_epochs: int = 2
