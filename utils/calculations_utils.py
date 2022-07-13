def calc_flop(train_config,model_config, max_input_size, embedding_size):
    B = train_config.batch_size
    s = max_input_size
    l = model_config.num_layers
    h = embedding_size
    V = model_config.vocab_size
    return 96 * B * s * l * h * h * (1 + s/6/h + V/16/l/h)