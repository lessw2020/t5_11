@dataclass
class train_config:
    batch_size: int = 128
    save_model: bool = True
    mixed_precision: bool = True
    host_port: str = "12399"
    activation_checkpointing: bool = True
    num_epochs: int = 2
    dataset_train = "datasets_grammar/gtrain_100k.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"
    model_checkpoint = "t5large_2e.pt"
