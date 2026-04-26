"""
计算预测结构与输入Domain结构的TMscore

工作流程：
1. 从TSV文件读取entry_id、domain序列列表和GT序列
2. 从domain序列反推在GT序列中的位置（domain_info）
3. 从GT结构中提取每个domain的结构片段
4. 处理<unk>：将断开的片段拼接并重新编号
5. 保存domain结构为PDB文件
6. 使用TMalign计算domain结构与设计结构的TMscore
7. 对每个样本，平均所有domain的TMscore
8. 保存结果到JSON
"""

import argparse
import json
import os
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

import biotite.structure.io as bsio
import numpy as np
from tqdm import tqdm

# TMalign 可执行文件路径
TMALIGN_EXEC = "/storage/yuanfajieLab/yuanfajie/my_project/analysis/structural_comparison/TMscore/TMalign"


def load_all_data_from_tsv(filepath, max_rows=-1):
    """
    从TSV文件加载所有数据

    参数:
    - filepath: TSV文件路径，格式为 entry_id, domain_list, gt_seqs, pred_seqs
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
        else:
            print(f"警告: TMalign输出格式异常: {pdb1} vs {pdb2}")
            tm_score = 0.0

        # 清理临时文件
        if os.path.exists(outpath):
            os.remove(outpath)

        return tm_score

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
    path = Path(gt_structure_dir) / f"AF-{entry_id}-F1-model_v4.cif"
    if path.exists():
        return str(path)

    return None


def domain_seq_to_domain_info(domain_seq, gt_seq):
    """
    从domain序列反推在GT序列中的位置

    参数:
    - domain_seq: domain序列，可能包含<unk>，如 "ABC<unk>DEF"
    - gt_seq: GT蛋白序列

    返回:
    - domain_info: domain位置信息，如 "10-12_50-52" (1-based索引)
    - 如果找不到，返回None
    """
    # 处理<unk>：分割成多个pieces
    pieces = [piece for piece in domain_seq.split("<unk>") if piece]

    if not pieces:
        return None

    # 查找每个piece在GT序列中的位置
    positions = []
    search_start = 0  # 从哪里开始搜索

    for piece in pieces:
        # 在GT序列中查找这个piece
        pos = gt_seq.find(piece, search_start)

        if pos == -1:
            # 找不到这个piece
            print(f"警告: 在GT {gt_seq}序列中找不到domain片段 '{piece}', domain 是 {domain_seq}")
            exit(1)

        # 记录位置 (1-based索引，左闭右闭区间)
        start = pos + 1
        end = pos + len(piece)
        positions.append(f"{start}-{end}")

        # 更新搜索起点，确保下一个piece在当前piece之后
        search_start = pos + len(piece)

    # 拼接成domain_info格式
    domain_info = "_".join(positions)
    return domain_info


def extract_domain_structure(gt_structure_path, domain_info, output_pdb_path):
    """
    从GT结构中提取domain片段并保存为PDB

    参数:
    - gt_structure_path: GT结构文件路径（CIF格式）
    - domain_info: domain信息，如 "10-50_80-120"
    - output_pdb_path: 输出PDB文件路径

    返回:
    - success: 是否成功
    """
    try:
        # 读取CIF结构
        structure = bsio.load_structure(gt_structure_path, model=1)

        # 解析domain_info，提取所有片段的残基范围
        ranges = []
        for pos_range in domain_info.split("_"):
            start, end = map(int, pos_range.split("-"))
            ranges.append((start, end))

        # 提取所有domain片段的原子
        all_atoms = []
        for start, end in ranges:
            # 选择这个范围内的原子
            mask = (structure.res_id >= start) & (structure.res_id <= end)
            fragment = structure[mask]
            all_atoms.append(fragment)

        # 拼接所有片段
        if len(all_atoms) == 1:
            combined_structure = all_atoms[0]
        else:
            # 使用biotite的+操作符拼接结构
            combined_structure = all_atoms[0]
            for fragment in all_atoms[1:]:
                combined_structure = combined_structure + fragment

        # 重新编号残基为连续的1, 2, 3...
        # 获取唯一的残基ID
        unique_res_ids = np.unique(combined_structure.res_id)
        # 创建映射：旧残基ID -> 新残基ID
        res_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_res_ids, start=1)}
        # 应用映射
        new_res_ids = np.array([res_id_mapping[old_id] for old_id in combined_structure.res_id])
        combined_structure.res_id = new_res_ids

        # 保存为PDB
        bsio.save_structure(output_pdb_path, combined_structure)

        return True

    except Exception as e:
        print(f"提取domain结构失败 {gt_structure_path}, domain {domain_info}: {e}")
        return False


def process_single_prediction(args):
    """
    处理单个预测的Domain TMscore计算

    参数:
    - args: (idx, entry_id, domain_seqs, gt_seq, pred_structure_path, gt_structure_dir, domain_temp_dir)

    返回:
    - (idx, mean_tm_score, domain_tm_scores): 索引、平均TMscore、所有domain的TMscore列表
    """
    idx, entry_id, domain_seqs, gt_seq, pred_structure_path, gt_structure_dir, domain_temp_dir = args

    # 查找GT结构
    gt_structure_path = find_gt_structure(entry_id, gt_structure_dir)

    if gt_structure_path is None:
        print(f"警告: 未找到 {entry_id} 的GT结构")
        return (idx, 0.0, [])

    # 检查预测结构是否存在
    if not os.path.exists(pred_structure_path):
        print(f"警告: 预测结构不存在: {pred_structure_path}")
        return (idx, 0.0, [])

    # 对每个domain计算TMscore
    domain_tm_scores = []

    for domain_idx, domain_seq in enumerate(domain_seqs):
        # 从domain序列反推domain_info
        domain_info = domain_seq_to_domain_info(domain_seq, gt_seq)

        if domain_info is None:
            print(f"警告: 无法从domain序列反推位置: {entry_id}, domain {domain_idx}")
            domain_tm_scores.append(0.0)
            continue

        # 为这个domain创建临时PDB文件
        domain_pdb_path = os.path.join(
            domain_temp_dir,
            f"{entry_id}_domain_{domain_idx}.pdb"
        )

        # 提取domain结构
        success = extract_domain_structure(gt_structure_path, domain_info, domain_pdb_path)

        if not success:
            print(f"警告: 提取domain结构失败: {entry_id}, domain {domain_idx}")
            domain_tm_scores.append(0.0)
            continue

        # 计算TMscore
        tm_score = calculate_tmscore(domain_pdb_path, pred_structure_path)
        domain_tm_scores.append(tm_score)

        # # 清理临时文件
        if os.path.exists(domain_pdb_path):
            os.remove(domain_pdb_path)

    # 计算平均TMscore
    if domain_tm_scores:
        mean_tm_score = np.mean(domain_tm_scores)
    else:
        mean_tm_score = 0.0

    return (idx, mean_tm_score, domain_tm_scores)


def parse_args():
    parser = argparse.ArgumentParser(
        description='计算预测结构与输入Domain结构的TMscore'
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
        '--domain_temp_dir',
        type=str,
        default='/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/analysis/AFDB_structure_download/0319-af_structures_domain_structures',
        help='临时domain结构保存目录'
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
    print("Domain TMscore 计算")
    print("="*60)
    print(f"测试输出路径: {args.test_output_path}")
    print(f"结构类型: {args.output_structure_type}")
    print(f"GT结构目录: {args.gt_structure_dir}")
    print(f"临时domain结构目录: {args.domain_temp_dir}")
    print(f"并行进程数: {args.num_processes}")
    print("="*60)

    # 创建临时目录
    Path(args.domain_temp_dir).mkdir(parents=True, exist_ok=True)

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
        pred_structure_paths = {
            line[0]: Path(pred_structure_dir) / f"pred_sequence_{line[0]}.pdb"
            for line in data_list
        }
    elif args.output_structure_type == "af3_output":
        pred_structure_dir = Path(args.test_output_path) / args.output_structure_type
        pred_structure_paths = {
            line[0]: Path(pred_structure_dir) / line[0] / f"{line[0]}_model.cif"
            for line in data_list
        }
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

        if entry_id is None:
            print(f"警告: 第 {idx} 行没有entry_id，跳过")
            continue

        if entry_id not in pred_structure_paths:
            print(f"警告: 未找到 {entry_id} 的预测结构")
            continue

        tasks.append((
            idx,
            entry_id,
            domains,      # domain序列列表
            gt_seq,       # GT蛋白序列
            str(pred_structure_paths[entry_id]),
            args.gt_structure_dir,
            args.domain_temp_dir
        ))

    print(f"\n准备计算 {len(tasks)} 个样本的Domain TMscore...")

    # 并行计算TMscore
    with Pool(processes=args.num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_prediction, tasks),
                total=len(tasks),
                desc="Calculating Domain TMscore"
            )
        )

    # 整理结果
    mean_tm_scores = {}
    all_domain_tm_scores = {}

    for idx, mean_tm_score, domain_tm_scores in results:
        entry_id = entry_id_list[idx]
        mean_tm_scores[entry_id] = mean_tm_score
        all_domain_tm_scores[entry_id] = domain_tm_scores

    # 计算统计
    overall_mean_tm_score = np.mean(list(mean_tm_scores.values()))

    print("\n" + "="*60)
    print("计算完成！")
    print("="*60)
    print(f"有效样本数量: {len(mean_tm_scores)}")
    print(f"平均Domain TMscore: {overall_mean_tm_score:.4f}")
    print("="*60)

    # 保存结果到JSON
    output_json = Path(args.test_output_path) / "log_metrics.json"

    if output_json.exists():
        with open(output_json, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics[f"Mean_TMscore_Domain_{args.output_structure_type}"] = float(overall_mean_tm_score)
    # metrics[f"TMscore_Domain_{args.output_structure_type}"] = mean_tm_scores
    metrics[f"TMscore_Domain_AllDomains_{args.output_structure_type}"] = all_domain_tm_scores

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n结果已保存到: {output_json}")


if __name__ == "__main__":
    main()
