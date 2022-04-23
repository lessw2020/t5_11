from dataclasses import dataclass


@dataclass
class train_config:
    batch_size: int = 8
    save_model: bool = True
    mixed_precision: bool = True
    host_port: str = "12369"
    activation_checkpointing: bool = True
