# This generating the sequence by domain conditioning

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.append(".")
from utils.config_utils import Config, load_config_from_yaml, merge_config
from utils.init_utils import construct_class_by_name


def main(args):
    # exp = _exp[args.training.experiment](args)
    exp = construct_class_by_name(**args.Trainer.kwargs.to_dict(), cfg=args)
    exp.start_generating()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path for configuration file", required=True
    )
    parser.add_argument(
        "--test_config",
        type=str,
        help="Path for test configuration file",
        required=True,
    )
    parser.add_argument(
        "--generation_config", type=str, default="configs/generation/default.yaml"
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()
    # args = config_utils._parse_args_and_yaml(parser)
    cfg = load_config_from_yaml(args.config)
    test_cfg = load_config_from_yaml(args.test_config)
    generation_cfg = load_config_from_yaml(args.generation_config)
    # merge args and cfg
    args = merge_config(cfg, args)
    args = merge_config(args, test_cfg)
    args = merge_config(args, generation_cfg)
    # Set the additional deterministic args
    args.Settings.run_name = os.path.basename(args.config).split(".")[0]
    args.Settings.num_nodes = int(os.environ.get("NUM_NODES", 1))
    args.Settings.num_gpus = torch.cuda.device_count() * args.Settings.num_nodes

    # 获取生成方法名称（如果配置中有的话）
    generation_method = getattr(args, "GenerationMethod", None)
    if generation_method and hasattr(generation_method, "method"):
        method_name = generation_method.method
        # 简化方法名用于目录命名
        if method_name == "generate_simple_with_unique_domain_pieces":
            method_suffix = "_unique"
        else:
            method_suffix = ""
    else:
        method_suffix = ""

    args.test_output_name = (
        os.path.basename(args.test_config).split(".")[0]
        + "_"
        + os.path.basename(args.generation_config).split(".")[0]
        + method_suffix
        + "_output"
    )

    # configure output dir
    args.istraining = False
    if args.ckpt_path:
        args.Settings.outdir = "/".join(args.ckpt_path.split("/")[:-2])
        args.test_output_dir = os.path.join(args.ckpt_path, args.test_output_name)
    else:
        args.Settings.outdir = os.path.join(args.Settings.outdir, "step0")
        args.test_output_dir = os.path.join(args.Settings.outdir, args.test_output_name)
    os.makedirs(args.test_output_dir, exist_ok=True)

    # Add environment variables to args
    args.environment = Config(dict(os.environ))
    if args.Settings.num_gpus == 1 or os.environ.get("LOCAL_RANK", "0") == "0":
        # write args to file
        log_path = os.path.join(args.test_output_dir, "generate_args.yaml")
        OmegaConf.save(args.to_dict(), f=log_path)
    main(args)
