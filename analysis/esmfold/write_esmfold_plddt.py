"""
This file is used to write the
1. global esmfold pLDDT
2. domain esmfold pLDDT -> if the domain is positioned well
3. linker esmfold pLDDT -> if the linker is reasonable
into the log_metrics.json file
"""

import argparse
import json
import os

import biotite.structure.io as bsio

os.environ["TORCH_HOME"] = "~/.cache"
import sys

import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.domain_matching.hard_domain_matching import find_best_domain_match


def get_pLDDT(pdb_file):
    """
    获取所有原子的 pLDDT 值
    """
    struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
    return struct.b_factor


def get_residue_plddt_by_position(struct, start_pos, end_pos):
    """
    获取指定位置范围内所有氨基酸的 pLDDT 值

    Args:
        pdb_file: PDB 文件路径
        start_pos: 起始位置（1-based）
        end_pos: 结束位置（1-based）

    Returns:
        numpy array: 该范围内所有原子的 pLDDT 值
    """

    # 获取 residue 信息
    residue_ids = struct.res_id
    # 找到指定范围内的原子
    atom_mask = (residue_ids >= start_pos) & (residue_ids <= end_pos)

    # 获取该范围内所有原子的 pLDDT 值
    region_plddt = struct.b_factor[atom_mask]

    return region_plddt


def load_all_data_from_tsv(filepath, max_rows: int = -1):
    """
    从TSV文件加载所有数据

    参数:
    - filepath: TSV文件路径，格式为 domain_list, gt_seqs, pred_seqs

    返回:
    - data_dict: 包含所有行的数据字典，key为entry_id，value为 (domains, gt_seq, pred_seq)
    """
    data_dict = {}

    with open(filepath, "r") as f:
        lines = f.readlines()[1:]  # skip the header
    if max_rows != -1:
        lines = lines[:max_rows]
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:  # 跳过空行
            continue

        parts = line.split("\t")
        entry_id = None
        if len(parts) == 4:
            entry_id = parts[0]
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
        except (IndexError, ValueError) as e:
            print(f"Error parsing tsv file {filepath}, line {line_idx + 1}")
            print("Error parts", parts)
            # continue
            raise e

        data_dict[entry_id if entry_id is not None else line_idx] = (
            domains,
            gt_seq,
            pred_seq,
        )

    return data_dict


def parse_domain_info_by_domain_piece_results(domain_piece_results, seq_len):
    """
    domain_piece_results: list of dict,
    which contains
    -  domain: domain_piece,
    -  domain_index: i,
    -  domain_subindex: j,
    -  match: {
        -  start: best_match.start(),
        -  end: best_match.end(),
        -  matched_sequence: best_match.group(),
    -  found: True
    """
    domain_regions = []
    linker_regions = []
    for domain_piece_result in domain_piece_results:
        if domain_piece_result["found"]:
            domain_regions.append(
                (
                    domain_piece_result["match"]["start"],
                    domain_piece_result["match"]["end"],
                )
            )
    domain_regions.sort(key=lambda x: x[0])
    # linker regions are the regions between the domain regions
    for i in range(len(domain_regions)):
        if i == 0:
            linker_regions.append((1, domain_regions[i][0] - 1))
        else:
            linker_regions.append(
                (domain_regions[i - 1][1] + 1, domain_regions[i][0] - 1)
            )
    if domain_regions:
        linker_regions.append((domain_regions[-1][1] + 1, seq_len))
    return domain_regions, linker_regions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    args = parser.parse_args()

    output_path = os.path.join(args.test_output_path, "esmfold_results")
    sequence_output_dict = load_all_data_from_tsv(
        os.path.join(args.test_output_path, "sequence_output.tsv")
    )
    # pae_dic, pLDDT_list = inference(model, cur_sequences_dict, output_path, save_file=True, verbose=accelerator.is_main_process, process_index=accelerator.process_index)
    pLDDT_dict = {}
    mean_plddt = []
    domain_plddt = []
    linker_plddt = []
    for file in tqdm(
        os.listdir(output_path),
        total=len(os.listdir(output_path)),
        desc="Calculating pLDDT",
    ):
        if file.endswith(".pdb"):
            # file name should be pred_sequence_{entry_id}.pdb
            entry_id = file.split(".")[0].split("_")[-1]
            if entry_id not in sequence_output_dict:
                entry_id = int(entry_id)

            domains, gt_seq, pred_seq = sequence_output_dict[entry_id]
            domain_piece_results, domain_results = find_best_domain_match(
                domains, pred_seq
            )
            domain_regions, linker_regions = parse_domain_info_by_domain_piece_results(
                domain_piece_results, len(pred_seq)
            )

            struct = bsio.load_structure(
                os.path.join(output_path, file), extra_fields=["b_factor"]
            )
            mean_plddt.append(struct.b_factor.mean())

            if domain_regions:
                for domain_region in domain_regions:
                    region_plddt = get_residue_plddt_by_position(
                        struct, domain_region[0], domain_region[1]
                    )
                    if len(region_plddt) > 0:
                        domain_plddt.append(region_plddt.mean())
            if linker_regions:
                for linker_region in linker_regions:
                    region_plddt = get_residue_plddt_by_position(
                        struct, linker_region[0], linker_region[1]
                    )
                    if region_plddt.size == 0:
                        continue
                    linker_plddt.append(region_plddt.mean())

    # write to log_metrics.json
    results_path = os.path.join(args.test_output_path, "log_metrics.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            result_dic = json.load(f)
    else:
        result_dic = {}

    if "ESMFold pLDDT" not in result_dic or result_dic["ESMFold pLDDT"] != np.nan:
        print("Writing ESMFold pLDDT")
        print(f"ESMFold pLDDT: {np.mean(mean_plddt)}")
        print(f"ESMFold domain pLDDT: {np.mean(domain_plddt)}")
        print(f"ESMFold linker pLDDT: {np.mean(linker_plddt)}")

        result_dic["ESMFold pLDDT"] = float(np.mean(mean_plddt))
        result_dic["ESMFold domain pLDDT"] = float(np.mean(domain_plddt))
        result_dic["ESMFold linker pLDDT"] = float(np.mean(linker_plddt))
        with open(results_path, "w") as f:
            json.dump(result_dic, f, indent=4)
