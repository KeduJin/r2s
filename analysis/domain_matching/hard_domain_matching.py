import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def visualize_domain_matching(
    domain_piece_results, domain_results, target_seq, save_path=None
):
    """
    可视化domain matching结果

    Args:
        domain_piece_results: domain匹配的详细结果
        domain_results: 每个domain是否找到的布尔列表
        target_seq: 目标序列
        pred_seq: 预测序列（可选，用于对比）
        save_path: 保存路径（可选）
    """

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建图形
    # fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig, axes = plt.subplots(1, 1, figsize=(12, 3))

    # 颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, len(domain_piece_results)))

    # 第一个子图：序列条带图
    ax1 = axes
    ax1.set_xlim(0, len(target_seq))
    ax1.set_ylim(-0.5, 1.5)

    # 绘制目标序列
    ax1.add_patch(
        Rectangle(
            (0, 0),
            len(target_seq),
            0.3,
            facecolor="lightgray",
            edgecolor="black",
            alpha=0.7,
        )
    )
    ax1.text(
        len(target_seq) / 2,
        0.15,
        f"Target Sequence (Length: {len(target_seq)})",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )

    # 绘制匹配的domain片段
    for i, result in enumerate(domain_piece_results):
        if result["found"]:
            start = result["match"]["start"]
            end = result["match"]["end"]
            domain_idx = result["domain_index"]
            sub_idx = result["domain_subindex"]

            # 使用不同颜色表示不同的domain
            color = colors[domain_idx]

            # 绘制匹配区域
            ax1.add_patch(
                Rectangle(
                    (start, 0.4),
                    end - start,
                    0.3,
                    facecolor=color,
                    edgecolor="black",
                    alpha=0.8,
                )
            )

            # 添加标签
            ax1.text(
                (start + end) / 2,
                0.55,
                f"D{domain_idx + 1}.{sub_idx + 1}",
                ha="center",
                va="center",
                fontsize=8,
                weight="bold",
            )

    ax1.set_xlabel("Sequence Position", fontsize=12)
    ax1.set_ylabel("Matching Region", fontsize=12)
    ax1.set_title("Domain Matching Visualization", fontsize=14, weight="bold")
    ax1.grid(True, alpha=0.3)

    # 添加图例
    legend_elements = []
    for i in range(len(domain_results)):
        color = colors[i]
        status = "Found" if domain_results[i] else "Not Found"
        legend_elements.append(
            mpatches.Patch(color=color, label=f"Domain {i + 1}: {status}")
        )
    ax1.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存到: {save_path}")

    plt.show()

    return fig


def find_best_domain_match(domain_list, target_seq):
    """
    找到每个domain的最佳匹配

    Args:
        domain_list: domain列表
        target_seq: 目标序列
        match_type: 'first' (第一个匹配) 或 'longest' (最长匹配)

    Returns:
        list: 每个domain的最佳匹配结果
    """
    domain_piece_results = []

    for i, domain in enumerate(domain_list):
        # domain_pattern = domain.replace("<unk>", ".*?")
        domain_pieces = domain.split("<unk>")
        for j, domain_piece in enumerate(domain_pieces):
            pattern = re.compile(domain_piece)

            matches = list(pattern.finditer(target_seq))

            if not matches:
                domain_piece_results.append(
                    {
                        "domain": domain_piece,
                        "domain_index": i,
                        "domain_subindex": j,
                        "match": None,
                        "found": False,
                    }
                )
                continue

            # if match_type == 'first':
            best_match = matches[0]
            # elif match_type == 'longest':
            #     best_match = max(matches, key=lambda m: len(m.group()))
            # else:
            #     best_match = matches[0]

            domain_piece_results.append(
                {
                    "domain": domain_piece,
                    "domain_index": i,
                    "domain_subindex": j,
                    "match": {
                        "start": best_match.start(),
                        "end": best_match.end(),
                        "matched_sequence": best_match.group(),
                        "length": len(best_match.group()),
                    },
                    "found": True,
                }
            )
    # domain_results only contain the domain is found or not
    domain_results = [True for i in range(len(domain_list))]
    for item in domain_piece_results:
        if not item["found"]:
            domain_results[item["domain_index"]] = False
    return domain_piece_results, domain_results


def print_match_results(results):
    """
    打印匹配结果
    """
    for result in results:
        print(
            f"\nDomain {result['domain_index'] + 1}, subindex {result['domain_subindex']}: {result['domain']}"
        )
        print(f"  位置: {result['match']['start']}-{result['match']['end']}")
        print(f"  匹配序列: {result['match']['matched_sequence']}")
