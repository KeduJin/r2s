
import blosum as bl
import numpy as np
import random
import numpy as np
import torch
from typing import Dict, Any

# 使用blosum库加载BLOSUM62矩阵
BLOSUM62_MATRIX = bl.BLOSUM(62)


def compute_perturbation_stats(batch: Dict[str, Any]) -> Dict[str, float]:
    """
    从batch中计算扰动统计信息（简化版）

    Args:
        batch: dataloader返回的batch字典，包含：
            - domain_boundary_changed: 是否发生边界变化 (batch_size,)
            - domain_boundary_change_ratio: domain边界变化导致的氨基酸位置变化比例 (batch_size,)
            - domain_mutation_ratio: domain序列的氨基酸变异比例 (batch_size,)

    Returns:
        dict: 统计信息字典，包含：
            - domain_boundary_changed_ratio_in_batch: batch中发生边界变化的样本比例
            - domain_boundary_change_ratio: 平均边界变化导致的氨基酸位置变化比例
            - domain_mutation_ratio: 平均domain序列变异比例
    """
    stats = {}

    # 1. domain边界变化统计
    if "domain_boundary_changed" in batch:
        changed = batch["domain_boundary_changed"]
        if isinstance(changed, torch.Tensor):
            changed = changed.cpu().numpy()
        elif isinstance(changed, list):
            changed = np.array(changed)

        if len(changed) > 0:
            # batch中发生边界变化的样本比例
            stats["domain_boundary_changed_ratio_in_batch"] = float(np.mean(changed))
        else:
            stats["domain_boundary_changed_ratio_in_batch"] = 0.0

    # 2. domain边界变化比例统计
    if "domain_boundary_change_ratio" in batch:
        ratios = batch["domain_boundary_change_ratio"]
        if isinstance(ratios, torch.Tensor):
            ratios = ratios.cpu().numpy()
        elif isinstance(ratios, list):
            ratios = np.array(ratios)

        if len(ratios) > 0:
            # 所有样本（包括未变化的）的平均变化比例
            stats["domain_boundary_change_ratio"] = float(np.mean(ratios))
        else:
            stats["domain_boundary_change_ratio"] = 0.0
    # 3. domain序列变异比例统计
    if "domain_mutation_ratio" in batch:
        ratios = batch["domain_mutation_ratio"]
        if isinstance(ratios, torch.Tensor):
            ratios = ratios.cpu().numpy()
        elif isinstance(ratios, list):
            ratios = np.array(ratios)

        if len(ratios) > 0:
            # batch中的平均domain序列变异比例
            stats["domain_mutation_ratio"] = float(np.mean(ratios))
        else:
            stats["domain_mutation_ratio"] = 0.0

    return stats



def sample_amino_acid_from_blosum(
    original_aa: str,
    temperature: float = 1.0,
    exclude_original: bool = True
) -> str:
    """
    根据BLOSUM62矩阵采样一个相似的氨基酸

    Args:
        original_aa: 原始氨基酸
        temperature: 温度参数，控制采样的保守程度
            - temperature < 1.0: 更保守，倾向于选择相似的氨基酸
            - temperature > 1.0: 更随机
        exclude_original: 是否排除原始氨基酸本身

    Returns:
        采样的氨基酸
    """
    # 标准氨基酸列表
    standard_amino_acids = list("ARNDCQEGHILKMFPSTWYV")

    # 如果不是标准氨基酸，返回原氨基酸
    if original_aa not in standard_amino_acids:
        return original_aa

    # 获取该氨基酸与所有氨基酸的BLOSUM得分
    score_values = []
    for aa in standard_amino_acids:
        # blosum库使用字典访问: matrix[aa1][aa2]
        try:
            score = BLOSUM62_MATRIX[original_aa][aa]
        except (KeyError, TypeError):
            # 如果找不到，使用默认值
            score = -4 if aa != original_aa else 1
        score_values.append(float(score))

    score_values = np.array(score_values, dtype=float)

    # 应用temperature
    score_values = score_values / temperature

    # Softmax转换为概率
    exp_scores = np.exp(score_values - np.max(score_values))  # 数值稳定性
    probabilities = exp_scores / np.sum(exp_scores)

    # 如果要排除原始氨基酸
    if exclude_original:
        original_idx = standard_amino_acids.index(original_aa)
        probabilities[original_idx] = 0
        probabilities = probabilities / np.sum(probabilities)  # 重新归一化

    # 采样
    sampled_aa = np.random.choice(standard_amino_acids, p=probabilities)
    return sampled_aa


def sample_amino_acid_from_random(
    original_aa: str,
    exclude_original: bool = True
) -> str:
    """
    完全随机采样一个氨基酸（作为BLOSUM采样的对照组）

    Args:
        original_aa: 原始氨基酸
        exclude_original: 是否排除原始氨基酸本身

    Returns:
        随机采样的氨基酸
    """
    # 标准氨基酸列表
    standard_amino_acids = list("ARNDCQEGHILKMFPSTWYV")

    # 如果不是标准氨基酸，返回原氨基酸
    if original_aa not in standard_amino_acids:
        return original_aa

    # 如果要排除原始氨基酸
    if exclude_original:
        candidate_aas = [aa for aa in standard_amino_acids if aa != original_aa]
    else:
        candidate_aas = standard_amino_acids

    # 均匀随机采样
    sampled_aa = random.choice(candidate_aas)
    return sampled_aa


def perturb_domain_sequence(
    seq: str,
    domain_info_list: list[str],
    mutation_rate: float = 0.05,
    boundary_mutation_rate_multiplier: float = 2.0,
    blosum_temperature: float = 1.0,
    boundary_residue_count: int = 5,
    use_blosum: bool = True,
) -> str:
    """
    扰动domain序列，模拟domain融合时的氨基酸变化

    Args:
        seq: 完整蛋白质序列
        domain_info_list: domain位置信息列表，格式如 ["1-100", "150-200_210-250"]
        mutation_rate: 基础变异率（domain内部区域）
        boundary_mutation_rate_multiplier: 边界区域变异率倍数
        blosum_temperature: BLOSUM采样温度（仅当use_blosum=True时有效）
        boundary_residue_count: 边界区域的氨基酸数量（从domain起始/结束算起）
        use_blosum: 是否使用BLOSUM矩阵采样，False则使用完全随机采样

    Returns:
        扰动后的序列
    """
    seq_list = list(seq)
    seq_len = len(seq)

    # 标记每个位置是否属于domain及其边界状态
    # 0: 非domain, 1: domain内部, 2: domain边界
    position_type = [0] * seq_len

    for domain_info in domain_info_list:
        # 解析domain位置（可能有多段，用_连接）
        pos_ranges = domain_info.split("_")

        for pos_range in pos_ranges:
            start, end = map(int, pos_range.split("-"))
            start_idx = start - 1  # 转为0-based索引
            end_idx = end  # end本身就是左闭右开的右边界

            # 标记domain内部
            for i in range(start_idx, end_idx):
                if i < seq_len:
                    position_type[i] = 1

            # 标记边界区域
            # 起始边界
            for i in range(start_idx, min(start_idx + boundary_residue_count, end_idx)):
                if i < seq_len:
                    position_type[i] = 2

            # 结束边界
            for i in range(max(end_idx - boundary_residue_count, start_idx), end_idx):
                if i < seq_len:
                    position_type[i] = 2

    # 对每个domain位置进行扰动
    for i in range(seq_len):
        if position_type[i] == 0:
            # 非domain区域，不扰动
            continue

        # 确定当前位置的变异率
        if position_type[i] == 2:
            # 边界区域
            current_rate = mutation_rate * boundary_mutation_rate_multiplier
        else:
            # 内部区域
            current_rate = mutation_rate

        # 根据变异率决定是否变异
        if random.random() < current_rate:
            original_aa = seq_list[i]
            # 根据use_blosum选择采样策略
            if use_blosum:
                # 使用BLOSUM矩阵采样新氨基酸
                new_aa = sample_amino_acid_from_blosum(
                    original_aa,
                    temperature=blosum_temperature,
                    exclude_original=True
                )
            else:
                # 使用完全随机采样
                new_aa = sample_amino_acid_from_random(
                    original_aa,
                    exclude_original=True
                )
            seq_list[i] = new_aa

    return ''.join(seq_list)


def perturb_domain(
    domain_info_list: list[str], seq: str, perturb_domain_max_offset: int = 5
) -> list[str]:
    """
    Perturb the domain information by adding or removing a random number of residues from the start or end of the domain.
    """
    new_domain_info_list = []
    for domain_info in domain_info_list:
        pos_ranges = domain_info.split("_")

        positions = []
        for pos_range in pos_ranges:
            start, end = map(int, pos_range.split("-"))
            positions.append([start, end])
        positions.sort(key=lambda x: x[0])

        delta = random.randint(1, perturb_domain_max_offset)
        op = random.choice(["add_start", "remove_start", "add_end", "remove_end"])

        if op == "add_start":
            positions[0][0] = max(1, positions[0][0] - delta)
        elif op == "remove_start":
            positions[0][0] = min(positions[0][1] - 1, positions[0][0] + delta)
        elif op == "add_end":
            positions[-1][1] = min(len(seq), positions[-1][1] + delta)
        elif op == "remove_end":
            positions[-1][1] = max(positions[-1][0] + 1, positions[-1][1] - delta)

        new_pos_ranges = [f"{start}-{end}" for start, end in positions]
        new_domain_info_list.append("_".join(new_pos_ranges))

    return new_domain_info_list