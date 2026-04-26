"""
将 satoken 与 sequence 两次生成的 sequence_output.tsv 整理为三个最终文件：

1. direct_satoken_generation.tsv  — 模型直接生成的 satoken
2. direct_sequence_generation.tsv — 模型直接生成的纯 sequence
3. sequence_from_satoken.tsv     — 从 (1) 中每行 satoken 提取的纯氨基酸序列

用法:
  python scripts/generate/merge_decoder_outputs.py \\
    --satoken_tsv <satoken 运行目录>/sequence_output.tsv \\
    --sequence_tsv <sequence 运行目录>/sequence_output.tsv \\
    --out_dir <输出目录>
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from utils.sequence_utils import extract_amino_acid_sequence_from_satoken  # noqa: E402


def _read_tsv_3col(path: str) -> list[list[str]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    if not rows:
        return []
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--satoken_tsv", required=True, help="satoken 生成得到的 sequence_output.tsv")
    p.add_argument("--sequence_tsv", required=True, help="sequence 生成得到的 sequence_output.tsv")
    p.add_argument("--out_dir", required=True, help="三个输出文件写入的目录")
    args = p.parse_args()

    for path in (args.satoken_tsv, args.sequence_tsv):
        if not os.path.isfile(path):
            raise SystemExit(f"文件不存在: {path}")

    os.makedirs(args.out_dir, exist_ok=True)

    s_rows = _read_tsv_3col(args.satoken_tsv)
    q_rows = _read_tsv_3col(args.sequence_tsv)
    if len(s_rows) < 2 or len(q_rows) < 2:
        raise SystemExit("至少需要一个表头行和一行数据")

    # 期望: reaction_smiles, raw_reaction, generated_seq
    sheader, sdata = s_rows[0], s_rows[1:]
    qheader, qdata = q_rows[0], q_rows[1:]

    for name, h in (("satoken", sheader), ("sequence", qheader)):
        if len(h) < 3:
            raise SystemExit(f"{name} tsv 列数不足 3: {h}")

    out_satoken = os.path.join(args.out_dir, "direct_satoken_generation.tsv")
    out_seq = os.path.join(args.out_dir, "direct_sequence_generation.tsv")
    out_from_s = os.path.join(args.out_dir, "sequence_from_satoken.tsv")

    with open(out_satoken, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow([sheader[0], sheader[1], "generated_satoken"])
        for row in sdata:
            if len(row) < 3:
                continue
            w.writerow([row[0], row[1], row[2]])

    with open(out_seq, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow([qheader[0], qheader[1], "generated_sequence"])
        for row in qdata:
            if len(row) < 3:
                continue
            w.writerow([row[0], row[1], row[2]])

    with open(out_from_s, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                sheader[0],
                sheader[1],
                "generated_satoken",
                "sequence_from_satoken",
            ]
        )
        for row in sdata:
            if len(row) < 3:
                continue
            smi, raw, sat = row[0], row[1], row[2]
            w.writerow(
                [
                    smi,
                    raw,
                    sat,
                    extract_amino_acid_sequence_from_satoken(sat),
                ]
            )

    print(
        f"已写入:\n  {out_satoken}\n  {out_seq}\n  {out_from_s}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
