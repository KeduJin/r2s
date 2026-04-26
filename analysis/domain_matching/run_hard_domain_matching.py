import argparse
import glob
import json
import os

import numpy as np
from hard_domain_matching import find_best_domain_match


def load_all_data_from_tsv(filepath, max_rows: int = -1):
    """
    从TSV文件加载所有数据

    参数:
    - filepath: TSV文件路径，格式为 domain_list, gt_seqs, pred_seqs

    返回:
    - data_list: 包含所有行的数据列表，每行为 (domains, gt_seq, pred_seq)
    """
    data_list = []

    with open(filepath, "r") as f:
        lines = f.readlines()[1:]  # skip the header
    if max_rows != -1:
        lines = lines[:max_rows]
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:  # 跳过空行
            continue

        parts = line.split("\t")
        if len(parts) == 4:
            domain_idx = 1
            gt_seq_idx = 2
            pred_seq_idx = 3
        elif len(parts) == 2:
            domain_idx = 0
            gt_seq_idx = None
            pred_seq_idx = 1

        domains_str = parts[domain_idx]
        # 移除方括号并分割
        domains_str = domains_str.strip("[]")
        domains = [d.strip().strip("'\"") for d in domains_str.split("', '")]
        try:
            # 获取ground truth序列（第二列）
            if gt_seq_idx is not None:
                gt_seq = parts[gt_seq_idx]
            else:
                gt_seq = None

            # 获取预测序列（第三列）
            pred_seq = parts[pred_seq_idx]
        except (IndexError, ValueError):
            print(f"Error parsing tsv file {filepath}, line {line_idx + 1}")
            print("Error parts", parts)
            continue

        data_list.append((domains, gt_seq, pred_seq))

    return data_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    tsv_file_list = glob.glob(os.path.join(args.test_output_path, "*.tsv"))
    mean_finded_ratio = []
    for tsv_file_path in tsv_file_list:
        data_list = load_all_data_from_tsv(tsv_file_path)
        for domains, gt_seq, pred_seq in data_list:
            domain_piece_results, domain_results = find_best_domain_match(
                domains, pred_seq
            )
            mean_finded_ratio.append(np.mean(domain_results))
    print(
        "{} mean_matched_domain_ratio, {:.2f}".format(
            args.test_output_path, np.mean(mean_finded_ratio)
        )
    )
    # tsv_file_res_dict[os.path.basename(tsv_file_path) + "-mean_finded_ratio"] = np.mean(domain_results)

    # mean_finded_ratio = tool.hard_analyze_tsv_file(tsv_file_path, )["mean_finded_ratio"]
    # tsv_file_res_dict[os.path.basename(tsv_file_path) + "-mean_finded_ratio"] = mean_finded_ratio

    # print(tsv_file_res_dict)
    # print(tsv_file_res_dict)
    # print(np.mean(list(tsv_file_res_dict.values())))
    # tsv_file_res_dict["overall_mean_seq_matching_ratio"] = np.mean(list(tsv_file_res_dict.values()))

    output_json = os.path.join(args.test_output_path, "log_metrics.json")
    res_dict = {}
    res_dict["mean_matched_domain_ratio"] = np.mean(mean_finded_ratio)
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            prev_tsv_file_res_dict = json.load(f)
        # res_dict.update(prev_tsv_file_res_dict)
        prev_tsv_file_res_dict.update(res_dict)
        res_dict = prev_tsv_file_res_dict

    with open(output_json, "w") as f:
        json.dump(res_dict, f)


if __name__ == "__main__":
    main()
