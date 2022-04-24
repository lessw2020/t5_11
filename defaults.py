from dataclasses import dataclass


@dataclass
class train_config:
    batch_size: int = 128
    save_model: bool = True
    mixed_precision: bool = True
    host_port: str = "12399"
    activation_checkpointing: bool = True
    num_epochs: int = 4
    dataset_train = "datasets_grammar/gtrain_500k.csv"
