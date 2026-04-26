"""
计算预测序列与Ground Truth结构的TMscore

工作流程：
1. 从TSV文件读取预测序列和对应的GT entry_id
2. 使用ESMFold折叠预测序列得到结构（或读取已有结构）
3. 找到对应的GT结构文件（从下载的AFDB结构中）
4. 使用TMalign计算TMscore
5. 保存结果到JSON
"""

import argparse
import json
import os
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

# TMalign 可执行文件路径
TMALIGN_EXEC = "/storage/yuanfajieLab/yuanfajie/my_project/analysis/structural_comparison/TMscore/TMalign"


def load_all_data_from_tsv(filepath, max_rows=-1):
    """
    从TSV文件加载所有数据

    参数:
    - filepath: TSV文件路径，格式为 domain_list, gt_seqs, pred_seqs 或 entry_id, domain_list, gt_seqs, pred_seqs
    - max_rows: 最大读取行数，-1表示读取所有

    返回:
    - data_list: 包含所有行的数据列表，每行为 (entry_id, domains, gt_seq, pred_seq)
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
            entry_id = parts[0]
            domain_idx = 1
            gt_seq_idx = 2
            pred_seq_idx = 3
        elif len(parts) == 3:
            # 格式: domain_list, gt_seq, pred_seq (没有entry_id)
            entry_id = None
            domain_idx = 0
            gt_seq_idx = 1
            pred_seq_idx = 2
        else:
            print(f"警告: 第{line_idx + 1}行格式错误，结束进程")
            exit(1)

        try:
            # 解析domains
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

        data_list.append((entry_id, domains, gt_seq, pred_seq))

    return data_list


def calculate_tmscore(pdb1, pdb2):
    """
    使用TMalign计算两个结构的TMscore

    参数:
    - pdb1: 第一个结构文件路径
    - pdb2: 第二个结构文件路径

    返回:
    - tm_score: TMscore值
    """
    random_id = random.randint(1, 1000000)
    outpath = f"/storage/yuanfajieLab/yuanfajie/tmpfile/tmp_tmalign_{random_id}.txt"
    # outpath = "tmp.txt"
    try:
        tmalign_cmd = f"{TMALIGN_EXEC} {pdb1} {pdb2} > {outpath}"
        subprocess.call(tmalign_cmd, shell=True, timeout=60)

        # 读取输出文件获取TMscore
        with open(outpath, 'r') as f:
            content = f.readlines()

        # TMalign输出的第14行包含TMscore (索引13)
        if len(content) > 13:
            q_target_line = content[13]
            q_tm_score = float(q_target_line.split()[1])
            t_target_line = content[14]
            t_tm_score = float(t_target_line.split()[1])
            tm_score = max(q_tm_score, t_tm_score)

            # seq_id = float(content[12].split()[-1])
        else:
            print(f"警告: TMalign输出格式异常: {pdb1} vs {pdb2}")
            tm_score = 0.0

        # 清理临时文件
        if os.path.exists(outpath):
            os.remove(outpath)

        return tm_score
        # return tm_score, seq_id

    except subprocess.TimeoutExpired:
        print(f"超时: {pdb1} vs {pdb2}")
        if os.path.exists(outpath):
            os.remove(outpath)
        return 0.0
    except Exception as e:
        print(f"计算TMscore失败: {e}")
        if os.path.exists(outpath):
            os.remove(outpath)
        return 0.0


def find_gt_structure(entry_id, gt_structure_dir):
    """
    根据entry_id查找GT结构文件

    参数:
    - entry_id: UniProt ID
    - gt_structure_dir: GT结构文件目录

    返回:
    - structure_path: 结构文件路径，如果未找到返回None
    """
    # 尝试不同的文件名格式

    path = Path(gt_structure_dir) / f"AF-{entry_id}-F1-model_v4.cif"
    if path.exists():
        return str(path)

    return None


def process_single_prediction(args):
    """
    处理单个预测的TMscore计算

    参数:
    - args: (idx, entry_id, pred_seq, pred_structure_path, gt_structure_dir)

    返回:
    - (idx, tm_score): 索引和TMscore值
    """
    idx, entry_id, pred_seq, pred_structure_path, gt_structure_dir = args

    # 查找GT结构
    gt_structure_path = find_gt_structure(entry_id, gt_structure_dir)

    if gt_structure_path is None:
        print(f"警告: 未找到 {entry_id} 的GT结构")
        return (idx, 0.0)

    # 检查预测结构是否存在
    if not os.path.exists(pred_structure_path):
        print(f"警告: 预测结构不存在: {pred_structure_path}")
        return (idx, 0.0)

    # 计算TMscore
    tm_score = calculate_tmscore(pred_structure_path, gt_structure_path)
    return (idx, tm_score)


def parse_args():
    parser = argparse.ArgumentParser(
        description='计算预测序列与GT结构的TMscore'
    )
    parser.add_argument(
        '--test_output_path',
        type=str,
        required=True,
        help='测试输出路径，包含TSV文件和预测结构'
    )
    parser.add_argument(
        '--output_structure_type',
        type=str,
        default='esmfold_results',
        choices=['esmfold_results', 'af3_output'],
        help='预测结构的类型'
    )
    parser.add_argument(
        '--gt_structure_dir',
        type=str,
        default='/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/analysis/AFDB_structure_download/0319-af_structures',
        help='GT结构文件目录'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=8,
        help='并行进程数'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=-1,
        help='最大处理行数，-1表示处理所有'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("Ground Truth TMscore 计算")
    print("="*60)
    print(f"测试输出路径: {args.test_output_path}")
    print(f"结构类型: {args.output_structure_type}")
    print(f"GT结构目录: {args.gt_structure_dir}")
    print(f"并行进程数: {args.num_processes}")
    print("="*60)

    # 查找TSV文件
    tsv_file_path = os.path.join(args.test_output_path, "sequence_output.tsv")
    if not os.path.exists(tsv_file_path):
        print(f"错误: TSV文件不存在: {tsv_file_path}")
        return

    # 加载数据
    print("\n加载TSV数据...")
    data_list = load_all_data_from_tsv(tsv_file_path, max_rows=args.max_rows)
    print(f"总共 {len(data_list)} 条数据")

    if not data_list:
        print("错误: 没有有效数据")
        return

    # 查找预测结构文件
    print("\n查找预测结构文件...")
    if args.output_structure_type == "esmfold_results":
        pred_structure_dir = Path(args.test_output_path) / args.output_structure_type
        # pred_sequence_
        pred_structure_paths = {line[0]: Path(pred_structure_dir) /f"pred_sequence_{line[0]}.pdb" for line in data_list}
        # pred_structure_paths = sorted(
        #     pred_structure_dir.glob("*.pdb"),
        #     key=lambda x: int(x.stem.split("_")[-1])
        # )
    elif args.output_structure_type == "af3_output":
        pred_structure_dir = Path(args.test_output_path) / args.output_structure_type
        pred_structure_paths = {line[0]: Path(pred_structure_dir) / line[0] /f"{line[0]}_model.cif" for line in data_list}
        # pred_structure_paths = sorted(
        #     pred_structure_dir.glob("*/*.cif"),
        #     key=lambda x: int(x.parent.name.split("_")[-1])
        # )
    else:
        print(f"错误: 不支持的结构类型: {args.output_structure_type}")
        return

    print(f"找到 {len(pred_structure_paths)} 个预测结构文件")

    if len(pred_structure_paths) != len(data_list):
        print(f"警告: 预测结构数量 ({len(pred_structure_paths)}) 与数据行数 ({len(data_list)}) 不匹配")

    # 准备任务
    tasks = []
    entry_id_list = []
    for idx, (entry_id, domains, gt_seq, pred_seq) in enumerate(data_list):
        entry_id_list.append(entry_id)
        if idx >= len(pred_structure_paths):
            print(f"警告: 索引 {idx} 超出预测结构范围")
            break

        if entry_id is None:
            print(f"警告: 第 {idx} 行没有entry_id，跳过")
            continue

        tasks.append((
            idx,
            entry_id,
            pred_seq,
            str(pred_structure_paths[entry_id]),
            args.gt_structure_dir
        ))

    print(f"\n准备计算 {len(tasks)} 个TMscore...")

    # 并行计算TMscore
    with Pool(processes=args.num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_prediction, tasks),
                total=len(tasks),
                desc="Calculating GT TMscore"
            )
        )

    # 整理结果
    tm_scores = {}
    # seq_ids = [0.0] * len(data_list)
    for idx, tm_score in results:
        tm_scores[entry_id_list[idx]] = tm_score


    # 计算统计
    mean_tm_score = np.mean(list(tm_scores.values()))
    # mean_tm_score = np.mean([s for s in tm_scores.values() if s > 0])
    # mean_seq_id = np.mean([s for s in seq_ids if s > 0])
    # valid_count = sum(1 for s in tm_scores if s > 0)

    print("\n" + "="*60)
    print("计算完成！")
    print("="*60)
    print(f"有效TMscore数量: {len(tm_scores)}")
    print(f"平均GT TMscore: {mean_tm_score:.4f}")
    # print(f"平均seq id: {mean_seq_id:.4f}")
    print("="*60)

    # 保存结果到JSON
    output_json = Path(args.test_output_path) / "log_metrics.json"

    if output_json.exists():
        with open(output_json, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics[f"Mean_TMscore_GT_{args.output_structure_type}"] = float(mean_tm_score)
    metrics[f"TMscore_GT_{args.output_structure_type}"] = tm_scores
    # metrics[f"Valid_TMscore_count_GT_{args.output_structure_type}"] = valid_count

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n结果已保存到: {output_json}")


if __name__ == "__main__":
    main()
