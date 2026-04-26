import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.append("/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain")
from utils.SequenceAlignment import SequenceAlignmentTool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    parser.add_argument("--min_similarity", type=float, default=0.5)
    parser.add_argument("--n_processes", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    tool = SequenceAlignmentTool()
    tsv_file_list = glob.glob(os.path.join(args.test_output_path, "*.tsv"))
    tsv_file_res_dict = {}
    for tsv_file_path in tsv_file_list:
        mean_seq_matching_ratio = tool.analyze_tsv_file_parallel(
            tsv_file_path,
            min_similarity=args.min_similarity,
            n_processes=args.n_processes,
            verbose=False,
        )["mean_seq_matching_ratio"]
        tsv_file_res_dict[
            os.path.basename(tsv_file_path) + "-mean_seq_matching_ratio"
        ] = mean_seq_matching_ratio

    tsv_file_res_dict["overall_mean_seq_matching_ratio"] = np.mean(
        list(tsv_file_res_dict.values())
    )

    output_json = os.path.join(args.test_output_path, "log_metrics.json")
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            prev_tsv_file_res_dict = json.load(f)
        tsv_file_res_dict.update(prev_tsv_file_res_dict)

    with open(output_json, "w") as f:
        json.dump(tsv_file_res_dict, f)


if __name__ == "__main__":
    main()
