"""
从文本构建 ReactionDataset 所需的 custom_reactions.pt。

每行一条反应，支持：
  - substrate>>product  （推荐，与 RDKit 反应用法一致，只按第一个 \">>\" 切分）
  - substrate<TAB>product

以 # 开头的行与空行忽略。

用法:
  python scripts/generate/build_custom_reactions_pt.py --out /path/to/custom_reactions.pt --from-file reactions.txt
  python scripts/generate/build_custom_reactions_pt.py --out custom_reactions.pt --smiles "CCO>>CC=O"
"""

from __future__ import annotations

import argparse
import torch


def _parse_line(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if ">>" in s:
        sub, prod = s.split(">>", 1)
        return sub.strip(), prod.strip()
    if "\t" in s:
        parts = s.split("\t", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return None


def _build_dict(lines: list[str], placeholder: str) -> dict:
    data: dict = {}
    n = 0
    for line in lines:
        pair = _parse_line(line)
        if pair is None:
            continue
        sub, prod = pair
        if not sub or not prod:
            raise ValueError(f"空 substrate/product: {line!r}")
        key = f"custom_{n}"
        data[key] = (sub, prod, placeholder)
        n += 1
    if not data:
        raise ValueError("未解析到任何反应行（支持 substrate>>product 或 substrate<TAB>product）")
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="输出 .pt 路径")
    p.add_argument(
        "--from-file",
        type=str,
        default=None,
        help="每行一条反应，substrate>>product 或 TSV 两列",
    )
    p.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="单条反应 substrate>>product，与 --from-file 互斥",
    )
    p.add_argument(
        "--placeholder-seq",
        type=str,
        default="A",
        help="三元组第三项占位（仅生成推理时常用，默认 A）",
    )
    args = p.parse_args()

    if args.from_file and args.smiles:
        raise SystemExit("请只指定 --from-file 或 --smiles 之一")
    if not args.from_file and not args.smiles:
        raise SystemExit("必须指定 --from-file 或 --smiles")

    if args.smiles:
        data = _build_dict([args.smiles], args.placeholder_seq)
    else:
        with open(args.from_file, "r", encoding="utf-8") as f:
            data = _build_dict(f.readlines(), args.placeholder_seq)

    torch.save(data, args.out)
    print(
        f"已保存 {len(data)} 条反应 -> {args.out}",
        f"(placeholder 第三项={args.placeholder_seq!r})",
        sep=" ",
    )


if __name__ == "__main__":
    main()
