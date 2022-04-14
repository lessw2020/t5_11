# main hq file for t5 training and prediction



def parse_args:
  parser = argparse.ArgumentParser(description="PyTorch fsdp T5.11 Example")
  parser.add_argument('--save-dir', default = '/model_chkpt',type=str
  parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
  args = parser.parse_args()
  return args



if name == __main__:
  
  args = parse_args()
  
  # seed
  torch.manual_seed(args.seed)
  gpus_per_node = torch.cuda.device_count()
 
  mp.spawn(fsdp_main, args=(gpus_per_node, args,), nprocs=WORLD_SIZE, join=True)
