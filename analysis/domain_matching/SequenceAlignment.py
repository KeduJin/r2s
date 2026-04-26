import multiprocessing as mp
import warnings
from difflib import SequenceMatcher
from functools import partial
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 忽略字体警告
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Glyph.*missing from font.*"
)


def load_data_from_tsv(filepath, row_index=0):
    """
    从TSV文件加载数据

    参数:
    - filepath: TSV文件路径
    - row_index: 要分析的行索引（从0开始）

    返回:
    - domains: domain列表
    - target_sequence: 目标序列
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    if row_index >= len(lines):
        raise ValueError(f"行索引 {row_index} 超出文件范围 (总共 {len(lines)} 行)")

    line = lines[row_index].strip()
    parts = line.split("\t")

    if len(parts) < 3:
        raise ValueError("TSV文件格式错误，每行应包含至少3列")

    # 解析domains（假设第一列是domains的字符串表示）
    domains_str = parts[0]
    # 移除方括号并分割
    domains_str = domains_str.strip("[]")
    domains = [d.strip().strip("'\"") for d in domains_str.split("', '")]

    # 获取目标序列（假设第二列是ground truth序列）
    target_sequence = parts[1]

    return domains, target_sequence, parts[2]


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
        if len(parts) < 3:
            print(f"警告: 第{line_idx + 1}行格式错误，跳过")
            continue

        # 解析domains（第一列）
        domains_str = parts[0]
        # 移除方括号并分割
        domains_str = domains_str.strip("[]")
        domains = [d.strip().strip("'\"") for d in domains_str.split("', '")]

        # 获取ground truth序列（第二列）
        gt_seq = parts[1]

        # 获取预测序列（第三列）
        pred_seq = parts[2]

        data_list.append((domains, gt_seq, pred_seq))

    return data_list


class SequenceAlignmentTool:
    def __init__(self):
        self.colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#98D8C8",
        ]

    def split_domain_with_unk(self, domain: str) -> List[str]:
        """将包含<unk>的domain分割成片段"""
        if "<unk>" not in domain:
            return [domain]

        # 按<unk>分割，保留<unk>标记
        parts = domain.split("<unk>")
        result = []

        for i, part in enumerate(parts):
            if part:  # 非空部分
                result.append(part)
            if i < len(parts) - 1:  # 不是最后一个部分
                result.append("<unk>")

        return result

    def find_unk_pattern_matches(
        self, target_seq: str, domain_parts: List[str]
    ) -> List[Tuple[int, int, float, List[Tuple[int, int, str]]]]:
        """查找带<unk>通配符的domain匹配，返回详细信息包括匹配片段"""
        if not domain_parts:
            return []

        matches = []

        # 递归查找匹配
        def find_matches_recursive(
            parts: List[str], start_pos: int, current_match: List[Tuple[int, int, str]]
        ) -> None:
            if not parts:
                # 所有部分都匹配完成，计算整体相似度
                if current_match:
                    total_matched_length = sum(
                        end - start for start, end, _ in current_match
                    )
                    total_domain_length = sum(
                        len(part) for part in domain_parts if part != "<unk>"
                    )
                    similarity = (
                        total_domain_length / total_matched_length
                        if total_matched_length > 0
                        else 0
                    )

                    # 计算匹配的起始和结束位置
                    match_start = current_match[0][0]
                    match_end = current_match[-1][1]
                    matches.append((match_start, match_end, similarity, current_match))
                return

            current_part = parts[0]
            remaining_parts = parts[1:]

            if current_part == "<unk>":
                # <unk>可以匹配任意长度的字符，尝试不同的长度
                max_unk_length = min(
                    50, len(target_seq) - start_pos
                )  # 限制<unk>最大长度

                for unk_length in range(0, max_unk_length + 1):
                    new_start = start_pos + unk_length
                    if new_start <= len(target_seq):
                        # 为<unk>部分添加标记
                        unk_match = (start_pos, new_start, "<unk>")
                        find_matches_recursive(
                            remaining_parts, new_start, current_match + [unk_match]
                        )
            else:
                # 查找当前部分的匹配
                pos = target_seq.find(current_part, start_pos)
                if pos != -1:
                    new_match = current_match + [
                        (pos, pos + len(current_part), current_part)
                    ]
                    find_matches_recursive(
                        remaining_parts, pos + len(current_part), new_match
                    )

        find_matches_recursive(domain_parts, 0, [])

        # 按相似度排序，返回最佳匹配
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:5]  # 返回前5个最佳匹配

    def find_exact_matches(
        self, target_seq: str, domains: List[str]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """查找完全匹配的domain位置"""
        matches = {}
        for i, domain in enumerate(domains):
            matches[f"domain_{i}"] = []
            start = 0
            while True:
                pos = target_seq.find(domain, start)
                if pos == -1:
                    break
                matches[f"domain_{i}"].append((pos, pos + len(domain)))
                start = pos + 1
        return matches

    def find_best_alignments(
        self, target_seq: str, domains: List[str], min_similarity: float = 0.6
    ) -> Dict[str, List[Tuple[int, int, float]]]:
        """使用滑动窗口找到最相似的序列片段，支持<unk>通配符"""
        alignments = {}

        for i, domain in enumerate(domains):
            alignments[f"domain_{i}"] = []

            # 检查是否包含<unk>
            if "<unk>" in domain:
                # 处理带<unk>的domain
                domain_parts = self.split_domain_with_unk(domain)
                unk_matches = self.find_unk_pattern_matches(target_seq, domain_parts)
                alignments[f"domain_{i}"] = unk_matches
            else:
                # 原有的滑动窗口匹配逻辑
                domain_len = len(domain)

                # 滑动窗口，窗口大小从domain长度的50%到150%
                min_window = max(1, int(domain_len * 0.5))
                max_window = min(len(target_seq), int(domain_len * 1.5))

                best_matches = []

                for window_size in range(min_window, max_window + 1):
                    for start in range(len(target_seq) - window_size + 1):
                        window_seq = target_seq[start : start + window_size]

                        # 计算相似度
                        similarity = SequenceMatcher(None, domain, window_seq).ratio()

                        if similarity >= min_similarity:
                            best_matches.append(
                                (start, start + window_size, similarity)
                            )

                # 按相似度排序，取前5个最佳匹配
                best_matches.sort(key=lambda x: x[2], reverse=True)
                alignments[f"domain_{i}"] = best_matches[:5]

        return alignments

    def calculate_domain_match_ratio(
        self, domains: List[str], pred_seq: str, min_similarity: float = 0.5
    ) -> Dict:
        """
        计算domain在预测序列中的匹配比例

        参数:
        - domains: domain列表
        - pred_seq: 预测序列
        - min_similarity: 最小相似度阈值

        返回:
        - 包含匹配统计信息的字典
        """
        alignments = self.find_best_alignments(pred_seq, domains, min_similarity)

        total_domains = len(domains)
        matched_domains = 0
        match_details = []

        for i, domain in enumerate(domains):
            domain_key = f"domain_{i}"
            domain_matches = alignments[domain_key]

            if domain_matches:
                # 获取最佳匹配
                best_match = domain_matches[0]
                if len(best_match) >= 3:
                    start, end, similarity = best_match[:3]
                    matched_domains += 1
                    match_details.append(
                        {
                            "domain_idx": i,
                            "domain": domain,
                            "matched": True,
                            "start": start,
                            "end": end,
                            "similarity": similarity,
                            "matched_seq": pred_seq[start:end],
                        }
                    )
                else:
                    match_details.append(
                        {
                            "domain_idx": i,
                            "domain": domain,
                            "matched": False,
                            "similarity": 0.0,
                        }
                    )
            else:
                match_details.append(
                    {
                        "domain_idx": i,
                        "domain": domain,
                        "matched": False,
                        "similarity": 0.0,
                    }
                )

        match_ratio = matched_domains / total_domains if total_domains > 0 else 0.0

        return {
            "total_domains": total_domains,
            "matched_domains": matched_domains,
            "match_ratio": match_ratio,
            "match_details": match_details,
        }

    def _process_single_row(self, row_data: Tuple, min_similarity: float) -> Dict:
        """
        处理单行数据的辅助函数，用于多进程并行处理

        参数:
        - row_data: (row_idx, domains, gt_seq, pred_seq) 元组
        - min_similarity: 最小相似度阈值

        返回:
        - 包含该行匹配结果的字典
        """
        row_idx, domains, gt_seq, pred_seq = row_data

        # 计算当前行的匹配情况
        match_result = self.calculate_domain_match_ratio(
            domains, pred_seq, min_similarity
        )

        return {
            "row_idx": row_idx,
            "domains": domains,
            "gt_seq": gt_seq,
            "pred_seq": pred_seq,
            "match_result": match_result,
        }

    def hard_analyze_tsv_file(
        self, filepath: str, max_rows: int = -1, verbose: bool = True
    ) -> Dict:
        data_list = load_all_data_from_tsv(filepath, max_rows)
        finded_ratio_list = []

        for row_idx, (domains, gt_seq, pred_seq) in enumerate(data_list):
            matched_domains = []
            for domain in domains:
                # 如果domain包含<unk>，需要特殊处理
                if "<unk>" in domain:
                    # 对于包含<unk>的domain，使用模式匹配
                    domain_parts = self.split_domain_with_unk(domain)
                    # 检查所有非<unk>部分是否都在pred_seq中
                    cur_domain_matched = True
                    for part in domain_parts:
                        if part != "<unk>" and part not in pred_seq:
                            cur_domain_matched = False
                            break
                else:
                    # 对于不包含<unk>的domain，直接检查是否在pred_seq中
                    cur_domain_matched = domain in pred_seq

                if cur_domain_matched:
                    matched_domains.append(domain)

            finded_ratio = len(matched_domains) / len(domains)
            finded_ratio_list.append(finded_ratio)

        return {
            "finded_ratio_list": finded_ratio_list,
            "mean_finded_ratio": np.mean(finded_ratio_list),
        }

    def analyze_tsv_file_parallel(
        self,
        filepath: str,
        min_similarity: float = 0.5,
        max_rows: int = -1,
        n_processes: int = None,
        verbose: bool = True,
    ) -> Dict:
        """
        使用多进程并行分析TSV文件中所有数据的domain匹配情况

        参数:
        - filepath: TSV文件路径
        - min_similarity: 最小相似度阈值
        - max_rows: 最大处理行数，-1表示处理所有行
        - n_processes: 进程数，None表示使用CPU核心数
        - verbose: 是否显示详细输出

        返回:
        - 包含整体统计信息的字典
        """
        data_list = load_all_data_from_tsv(filepath, max_rows)
        total_rows = len(data_list)

        if n_processes is None:
            n_processes = min(mp.cpu_count(), total_rows)  # 不超过数据行数

        if verbose:
            print(f"开始并行分析TSV文件: {filepath}")
            print(f"总共 {total_rows} 行数据")
            print(f"使用 {n_processes} 个进程")
            print("=" * 60)

        # 准备数据：将行索引添加到每行数据中
        row_data_list = [
            (row_idx, domains, gt_seq, pred_seq)
            for row_idx, (domains, gt_seq, pred_seq) in enumerate(data_list)
        ]

        # 创建处理函数（绑定min_similarity参数）
        process_func = partial(self._process_single_row, min_similarity=min_similarity)

        # 使用多进程池处理数据
        with mp.Pool(processes=n_processes) as pool:
            # if verbose:
            # print("正在并行处理数据...")
            row_results = list(
                tqdm(
                    pool.imap(process_func, row_data_list),
                    total=total_rows,
                    desc="Processing file: " + filepath,
                )
            )
            # else:
            #     row_results = pool.map(process_func, row_data_list)

        # 按行索引排序，确保结果顺序正确
        row_results.sort(key=lambda x: x["row_idx"])

        # 计算整体统计
        total_domains = 0
        total_matched_domains = 0

        for row_result in row_results:
            match_result = row_result["match_result"]
            total_domains += match_result["total_domains"]
            total_matched_domains += match_result["matched_domains"]

        overall_match_ratio = (
            total_matched_domains / total_domains if total_domains > 0 else 0.0
        )
        mean_seq_matching_ratio = np.mean(
            [row["match_result"]["match_ratio"] for row in row_results]
        )

        if verbose:
            print("\n" + "=" * 60)
            print("整体统计结果:")
            print(f"总行数: {total_rows}")
            print(f"总domain数: {total_domains}")
            print(f"匹配的domain数: {total_matched_domains}")
            print(f"整体匹配比例: {overall_match_ratio:.3f}")
            print(f"平均序列匹配比例: {mean_seq_matching_ratio:.3f}")
            print(f"使用进程数: {n_processes}")

        return {
            "total_rows": total_rows,
            "total_domains": total_domains,
            "total_matched_domains": total_matched_domains,
            "overall_match_ratio": overall_match_ratio,
            "mean_seq_matching_ratio": mean_seq_matching_ratio,
            "row_results": row_results,
            "n_processes_used": n_processes,
        }

    # def analyze_tsv_file_chunked_parallel(self, filepath: str, min_similarity: float = 0.5,
    #                                      max_rows: int = -1, n_processes: int = None,
    #                                      chunk_size: int = 100, verbose: bool = True) -> Dict:
    #     """
    #     使用分块并行处理分析TSV文件，适合处理非常大的文件

    #     参数:
    #     - filepath: TSV文件路径
    #     - min_similarity: 最小相似度阈值
    #     - max_rows: 最大处理行数，-1表示处理所有行
    #     - n_processes: 进程数，None表示使用CPU核心数
    #     - chunk_size: 每个进程处理的数据块大小
    #     - verbose: 是否显示详细输出

    #     返回:
    #     - 包含整体统计信息的字典
    #     """
    #     data_list = load_all_data_from_tsv(filepath, max_rows)
    #     total_rows = len(data_list)

    #     if n_processes is None:
    #         n_processes = min(mp.cpu_count(), (total_rows + chunk_size - 1) // chunk_size)

    #     if verbose:
    #         print(f"开始分块并行分析TSV文件: {filepath}")
    #         print(f"总共 {total_rows} 行数据")
    #         print(f"使用 {n_processes} 个进程，每块 {chunk_size} 行")
    #         print("="*60)

    #     # 将数据分块
    #     chunks = []
    #     for i in range(0, total_rows, chunk_size):
    #         chunk_data = []
    #         for j in range(i, min(i + chunk_size, total_rows)):
    #             domains, gt_seq, pred_seq = data_list[j]
    #             chunk_data.append((j, domains, gt_seq, pred_seq))
    #         chunks.append(chunk_data)

    #     def process_chunk(chunk_data: List[Tuple], min_sim: float) -> List[Dict]:
    #         """处理一个数据块"""
    #         chunk_results = []
    #         for row_data in chunk_data:
    #             result = self._process_single_row(row_data, min_sim)
    #             chunk_results.append(result)
    #         return chunk_results

    #     # 创建处理函数
    #     process_func = partial(process_chunk, min_sim=min_similarity)

    #     # 使用多进程池处理数据块
    #     with mp.Pool(processes=n_processes) as pool:
    #         if verbose:
    #             print("正在并行处理数据块...")
    #             chunk_results = list(tqdm(
    #                 pool.imap(process_func, chunks),
    #                 total=len(chunks),
    #                 desc="Processing chunks"
    #             ))
    #         else:
    #             chunk_results = pool.map(process_func, chunks)

    #     # 合并所有块的结果
    #     row_results = []
    #     for chunk_result in chunk_results:
    #         row_results.extend(chunk_result)

    #     # 按行索引排序
    #     row_results.sort(key=lambda x: x['row_idx'])

    #     # 计算整体统计
    #     total_domains = 0
    #     total_matched_domains = 0

    #     for row_result in row_results:
    #         match_result = row_result['match_result']
    #         total_domains += match_result['total_domains']
    #         total_matched_domains += match_result['matched_domains']

    #     overall_match_ratio = total_matched_domains / total_domains if total_domains > 0 else 0.0
    #     mean_seq_matching_ratio = np.mean([row['match_result']['match_ratio'] for row in row_results])

    #     if verbose:
    #         print("\n" + "="*60)
    #         print("整体统计结果:")
    #         print(f"总行数: {total_rows}")
    #         print(f"总domain数: {total_domains}")
    #         print(f"匹配的domain数: {total_matched_domains}")
    #         print(f"整体匹配比例: {overall_match_ratio:.3f}")
    #         print(f"平均序列匹配比例: {mean_seq_matching_ratio:.3f}")
    #         print(f"使用进程数: {n_processes}")
    #         print(f"数据块数: {len(chunks)}")

    #     return {
    #         'total_rows': total_rows,
    #         'total_domains': total_domains,
    #         'total_matched_domains': total_matched_domains,
    #         'overall_match_ratio': overall_match_ratio,
    #         'mean_seq_matching_ratio': mean_seq_matching_ratio,
    #         'row_results': row_results,
    #         'n_processes_used': n_processes,
    #         'chunk_size': chunk_size,
    #         'n_chunks': len(chunks)
    #     }

    def highlight_sequence(
        self, target_seq: str, matches: Dict, alignments: Dict, domains: List[str]
    ) -> str:
        """生成高亮的序列字符串"""
        # 创建位置到颜色的映射
        position_colors = {}

        # 处理完全匹配
        for domain_key, positions in matches.items():
            color_idx = int(domain_key.split("_")[1])
            color = self.colors[color_idx % len(self.colors)]
            for start, end in positions:
                for pos in range(start, end):
                    position_colors[pos] = color

        # 处理相似匹配（如果没有完全匹配）
        for domain_key, alignments_list in alignments.items():
            if not matches.get(domain_key):  # 只有在没有完全匹配时才使用相似匹配
                color_idx = int(domain_key.split("_")[1])
                color = self.colors[color_idx % len(self.colors)]
                for start, end, similarity in alignments_list:
                    for pos in range(start, end):
                        if pos not in position_colors:  # 避免覆盖完全匹配
                            position_colors[pos] = color

        # 生成高亮序列
        highlighted = []
        for i, char in enumerate(target_seq):
            if i in position_colors:
                highlighted.append(
                    f"<span style='background-color: {position_colors[i]}; color: white; font-weight: bold;'>{char}</span>"
                )
            else:
                highlighted.append(char)

        return "".join(highlighted)

    def create_visualization(
        self, target_seq: str, alignments: Dict, domains: List[str]
    ):
        """创建可视化图表"""
        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 3))

        # 上图：序列条带图
        y_pos = 0
        seq_len = len(target_seq)

        # 绘制目标序列背景
        ax1.add_patch(
            patches.Rectangle(
                (0, y_pos - 0.4),
                seq_len,
                0.8,
                facecolor="lightgray",
                edgecolor="black",
                alpha=0.3,
            )
        )

        # 绘制相似匹配 - 只显示最佳匹配
        for domain_key, alignments_list in alignments.items():
            if alignments_list:  # 确保有匹配结果
                color_idx = int(domain_key.split("_")[1])
                color = self.colors[color_idx % len(self.colors)]

                # 只绘制第一个（最佳）匹配
                match_info = alignments_list[0]

                # 检查这个domain是否包含<unk>
                domain_idx = int(domain_key.split("_")[1])
                domain = domains[domain_idx]

                if "<unk>" in domain and len(match_info) > 3:
                    # 对于包含<unk>的domain，分别绘制确定匹配部分和<unk>部分
                    start, end, similarity, match_parts = match_info

                    for part_start, part_end, part_type in match_parts:
                        if part_type == "<unk>":
                            # <unk>部分用灰色
                            ax1.add_patch(
                                patches.Rectangle(
                                    (part_start, y_pos - 0.4),
                                    part_end - part_start,
                                    0.8,
                                    facecolor="gray",
                                    edgecolor="black",
                                    alpha=0.5,
                                )
                            )
                        else:
                            # 确定匹配部分用原色
                            ax1.add_patch(
                                patches.Rectangle(
                                    (part_start, y_pos - 0.4),
                                    part_end - part_start,
                                    0.8,
                                    facecolor=color,
                                    edgecolor="black",
                                    alpha=0.5,
                                )
                            )
                else:
                    # 普通匹配，直接绘制整个区域
                    start, end, similarity = match_info[:3]
                    ax1.add_patch(
                        patches.Rectangle(
                            (start, y_pos - 0.4),
                            end - start,
                            0.8,
                            facecolor=color,
                            edgecolor="black",
                            alpha=0.5,
                        )
                    )

                # 添加位置标注和相似度
                start, end, similarity = match_info[:3]
                ax1.text(
                    start + (end - start) / 2,
                    y_pos - 0.7,
                    f"{start}-{end}\n({similarity:.2f})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

        ax1.set_xlim(0, seq_len)
        ax1.set_ylim(-1, 1)
        ax1.set_xlabel("Sequence Position")
        ax1.set_title("Sequence Alignment Result")
        ax1.set_yticks([])

        # 添加图例
        legend_elements = []
        for i, domain in enumerate(domains):
            color = self.colors[i % len(self.colors)]
            # 显示domain的前20个字符，如果包含<unk>则标注
            display_domain = domain[:20] + ("..." if len(domain) > 20 else "")
            if "<unk>" in domain:
                display_domain += " (with <unk>)"
            legend_elements.append(
                patches.Patch(color=color, label=f"Domain {i + 1}: {display_domain}")
            )

        # 添加<unk>的图例说明
        legend_elements.append(patches.Patch(color="gray", label="<unk> (wildcard)"))
        ax1.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        return fig

    def analyze_sequence(
        self,
        target_seq: str,
        domains: List[str],
        min_similarity: float = 0.6,
        plot: bool = True,
    ):
        """完整的序列分析流程"""
        print(f"目标序列长度: {len(target_seq)}")
        print(f"Domain数量: {len(domains)}")
        print("\n" + "=" * 50)

        # 查找相似匹配（现在支持<unk>通配符）
        similar_alignments = self.find_best_alignments(
            target_seq, domains, min_similarity
        )

        # 打印结果
        for i, domain in enumerate(domains):
            domain_key = f"domain_{i}"
            print(f"\nDomain {i + 1}: {domain}")
            print(f"长度: {len(domain)}")

            if "<unk>" in domain:
                print("包含<unk>通配符，已进行模式匹配")
                domain_parts = self.split_domain_with_unk(domain)
                print(f"分割后的片段: {domain_parts}")

            if similar_alignments[domain_key]:
                print("最相似的匹配:")
                start, end, similarity = similar_alignments[domain_key][0][:3]
                print(f"  {start}-{end}, 相似度: {similarity:.3f}")
                print(f"     匹配序列: {target_seq[start:end]}")
            else:
                print("  未找到匹配")

        # 创建可视化
        if plot:
            fig = self.create_visualization(target_seq, similar_alignments, domains)
        else:
            fig = None

        return similar_alignments, fig


# 使用示例
if __name__ == "__main__":
    # 创建分析工具实例
    tool = SequenceAlignmentTool()

    # 使用并行版本进行分析
    result = tool.analyze_tsv_file_parallel(
        "output/2025Y_12D_09M_17h-gpt/IntervalCheckpoints/step=50000/test_output_multinomial/sequence_output_rank0.tsv",
        min_similarity=0.5,
        max_rows=100,
        n_processes=64,  # 使用4个进程
        verbose=True,
    )

    # 或者使用分块并行版本（适合大文件）
    # result = tool.analyze_tsv_file_chunked_parallel(
    #     "your_file.tsv",
    #     min_similarity=0.5,
    #     max_rows=1000,
    #     n_processes=8,
    #     chunk_size=50,
    #     verbose=True
    # )
