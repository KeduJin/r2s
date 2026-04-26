"""
计算预测序列与ground truth序列的sequence identity

使用标准的全局比对方式计算seq_id:
seq_id = 匹配的氨基酸数 / 比对长度
"""

import argparse
import glob
import json
import os

import numpy as np
from Bio import Align
from tqdm import tqdm


def calculate_sequence_identity(seq1, seq2):
    """
    使用Bio.Align计算两个序列的sequence identity
    使用全局比对，标准的seq_id计算方式

    参数:
    - seq1: 第一个序列（通常是GT序列）
    - seq2: 第二个序列（通常是预测序列）

    返回:
    - identity_pct: 序列一致性百分比 (0-1之间的浮点数)
    """
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0

    # 创建比对器
    aligner = Align.PairwiseAligner()

    # 使用全局比对（默认模式）
    aligner.mode = 'global'

    # 进行比对
    alignments = aligner.align(seq1, seq2)

    try:
        best_alignment = alignments[0]
    except IndexError:
        return 0.0

    # 提取比对信息
    counts = best_alignment.counts()
    matches = counts.identities
    # 比对长度 = 匹配 + 错配 + 所有空位
    alignment_length = counts.aligned + counts.gaps

    identity_pct = (matches / alignment_length) if alignment_length > 0 else 0.0

    return identity_pct


def load_all_data_from_tsv(filepath, max_rows=-1):
    """
    从TSV文件加载所有数据

    参数:
    - filepath: TSV文件路径，格式为 domain_list, gt_seqs, pred_seqs
    - max_rows: 最大读取行数，-1表示读取所有

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

        # 判断文件格式
        if len(parts) == 4:
            # 格式: entry_id, domain_list, gt_seq, pred_seq
            domain_idx = 1
            gt_seq_idx = 2
            pred_seq_idx = 3
        elif len(parts) == 3:
            # 格式: domain_list, gt_seq, pred_seq
            domain_idx = 0
            gt_seq_idx = 1
            pred_seq_idx = 2
        elif len(parts) == 2:
            # 格式: domain_list, pred_seq (没有gt_seq)
            print(f"警告: 第{line_idx + 1}行没有GT序列，跳过")
            exit(1)
        else:
            print(f"警告: 第{line_idx + 1}行格式错误，跳过")
            exit(1)

        try:
            # 解析domains（第一列或第二列）
            domains_str = parts[domain_idx]
            domains_str = domains_str.strip("[]")
            domains = [d.strip().strip("'\"") for d in domains_str.split("', '")]

            # 获取ground truth序列
            gt_seq = parts[gt_seq_idx]

            # 获取预测序列
            pred_seq = parts[pred_seq_idx]

        except (IndexError, ValueError) as e:
            print(f"Error parsing tsv file {filepath}, line {line_idx + 1}: {e}")
            print("Error parts", parts)
            continue

        data_list.append((domains, gt_seq, pred_seq))

    return data_list


def parse_args():
    parser = argparse.ArgumentParser(
        description='计算预测序列与GT序列的sequence identity（标准全局比对方式）'
    )
    parser.add_argument("--test_output_path", type=str, required=True,
                       help="测试输出路径，包含TSV文件")
    parser.add_argument("--max_rows", type=int, default=-1,
                       help="最大处理行数，-1表示处理所有行")
    parser.add_argument("--verbose", action='store_true',
                       help="显示详细输出")
    return parser.parse_args()


def main():
    args = parse_args()

    # 查找所有TSV文件
    # tsv_file_list = glob.glob(os.path.join(args.test_output_path, "*.tsv"))
    tsv_file_path = os.path.join(args.test_output_path, "sequence_output.tsv")

    if not os.path.exists(tsv_file_path):
        print(f"警告: 在 {args.test_output_path} 中未找到TSV文件")
        return

    # 用于汇总所有文件的seq_id
    all_seq_ids = []

    # 处理每个TSV文件
    # for tsv_file_path in tsv_file_list:
    if args.verbose:
        print(f"\n处理文件: {os.path.basename(tsv_file_path)}")

    data_list = load_all_data_from_tsv(tsv_file_path, max_rows=args.max_rows)

    if not data_list:
        print(f"警告: {tsv_file_path} 中没有有效数据")
        exit(1)

    # 对每一行计算seq_id
    file_seq_ids = []

    for domains, gt_seq, pred_seq in tqdm(
        data_list,
        desc=f"Processing {os.path.basename(tsv_file_path)}",
        disable=not args.verbose
    ):
        seq_id = calculate_sequence_identity(gt_seq, pred_seq)
        file_seq_ids.append(seq_id)

    # 汇总当前文件的结果
    all_seq_ids.extend(file_seq_ids)

    if args.verbose:
        print(f"  文件平均 GT seq_id: {np.mean(file_seq_ids):.4f}")

    # 计算整体平均指标
    overall_mean_seqid = np.mean(all_seq_ids) if all_seq_ids else 0.0

    # 打印结果
    print("\n" + "="*60)
    print(f"测试输出路径: {args.test_output_path}")
    print(f"处理的样本总数: {len(all_seq_ids)}")
    print(f"\n整体平均 GT sequence identity: {overall_mean_seqid:.4f} ({overall_mean_seqid:.2%})")
    print("="*60)

    # 保存结果到JSON
    output_json = os.path.join(args.test_output_path, "log_metrics.json")
    res_dict = {
        'mean_gt_seqid': overall_mean_seqid,
    }

    # 如果已存在log_metrics.json，合并结果
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            prev_metrics = json.load(f)
        prev_metrics.update(res_dict)
        res_dict = prev_metrics

    with open(output_json, "w") as f:
        json.dump(res_dict, f, indent=2)

    print(f"\n结果已保存到: {output_json}")


if __name__ == "__main__":
    main()
