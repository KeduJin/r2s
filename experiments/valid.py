import argparse
import os
import sys
from datetime import datetime

import torch
from omegaconf import OmegaConf

sys.path.append(".")
from utils.config_utils import Config, load_config_from_yaml, merge_config
from utils.init_utils import construct_class_by_name


def main(args):
    # exp = _exp[args.training.experiment](args)
    exp = construct_class_by_name(**args.Trainer.kwargs.to_dict(), cfg=args)
    exp.start_validation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path for configuration file", required=True
    )
    parser.add_argument("--continue_training", type=str, default=None)
    parser.add_argument("--not_load_optim", action="store_true", default=False)
    args = parser.parse_args()
    args.not_load_optim = True  # in valid, we do not need to load the optimizer
    # args = config_utils._parse_args_and_yaml(parser)
    cfg = load_config_from_yaml(args.config)
    # merge args and cfg
    args = merge_config(cfg, args)
    # Set the additional deterministic args
    args.Settings.run_name = os.path.basename(args.config).split(".")[0]
    args.Settings.num_nodes = int(os.environ.get("NUM_NODES", 1))
    args.Settings.num_gpus = torch.cuda.device_count() * args.Settings.num_nodes
    args.Model.Scheduler_kwargs.num_warmup_steps *= args.Settings.num_gpus
    args.Model.Scheduler_kwargs.num_training_steps *= args.Settings.num_gpus

    # configure output dir
    args.istraining = True
    if args.continue_training:
        args.Settings.outdir = "/".join(args.continue_training.split("/")[:-2])
    else:
        dt_string = datetime.now().strftime("%YY_%dD_%mM_%Hh")
        args.Settings.outdir = os.path.join(
            args.Settings.outdir, dt_string + "-" + args.Settings.run_name
        )
        os.makedirs(args.Settings.outdir, exist_ok=True)

    # Add environment variables to args
    args.environment = Config(dict(os.environ))
    if args.Settings.num_gpus == 1 or os.environ.get("LOCAL_RANK", "0") == "0":
        # write args to file
        log_path = os.path.join(args.Settings.outdir, "args_validation.yaml")
        OmegaConf.save(args.to_dict(), f=log_path)

    main(args)
