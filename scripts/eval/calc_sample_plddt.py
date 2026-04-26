import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import biotite.structure.io as bsio
import esm
import numpy as np
import torch


def read_sequences_from_tsv(tsv_path: str, seq_column: str) -> Dict[str, str]:
    sequences = {}
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entry_id = row["entry_id"]
            seq = row[seq_column].strip()
            if seq:
                sequences[entry_id] = seq
    return sequences


def calc_plddt_from_pdb(pdb_path: str) -> float:
    struct = bsio.load_structure(pdb_path, extra_fields=["b_factor"])
    return float(struct.b_factor.mean())


def calc_plddt_from_pdb_str(pdb_str: str) -> float:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=True, encoding="utf-8"
    ) as tmp:
        tmp.write(pdb_str)
        tmp.flush()
        return calc_plddt_from_pdb(tmp.name)


def run_one_set(
    model,
    sequences: Dict[str, str],
    out_dir: Path,
    label: str,
    *,
    batch_size: int,
    plddt_from: str,
    save_pdbs: bool,
    num_recycles: Optional[int],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Tuple[str, str]] = list(sequences.items())
    n_total = len(items)
    per_seq: List[dict] = []
    n_printed = 0

    for start in range(0, n_total, batch_size):
        batch = items[start : start + batch_size]
        entry_ids = [eid for eid, _ in batch]
        seqs_full = [seq for _, seq in batch]
        try:
            with torch.inference_mode():
                seqs_trunc = [s[:1024] for s in seqs_full]
                if plddt_from == "network":
                    out = model.infer(
                        seqs_trunc,
                        num_recycles=num_recycles,
                    )
                    plddt_vals = (
                        out["mean_plddt"].detach().float().cpu().reshape(-1).tolist()
                    )
                    pdb_strs: Optional[List[str]] = None
                    if save_pdbs:
                        pdb_strs = model.output_to_pdb(out)
                else:
                    pdb_strs = model.infer_pdbs(
                        seqs_trunc,
                        num_recycles=num_recycles,
                    )
                    plddt_vals = [calc_plddt_from_pdb_str(s) for s in pdb_strs]

                for j, entry_id in enumerate(entry_ids):
                    plddt = float(plddt_vals[j])
                    if save_pdbs and pdb_strs is not None:
                        pdb_path = out_dir / f"{label}_{entry_id}.pdb"
                        with open(pdb_path, "w", encoding="utf-8") as f:
                            f.write(pdb_strs[j])
                    per_seq.append(
                        {
                            "entry_id": entry_id,
                            "seq_len": len(sequences[entry_id]),
                            "plddt": plddt,
                        }
                    )
                    n_printed += 1
                    print(
                        f"[{label}] {n_printed}/{n_total} {entry_id} pLDDT={plddt:.2f}"
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(
                f"[{label}] batch 失败 (size={len(batch)}): {e}，改逐条重试"
            )
            for entry_id, seq in batch:
                try:
                    with torch.inference_mode():
                        st = [seq[:1024]]
                        if plddt_from == "network":
                            out = model.infer(
                                st,
                                num_recycles=num_recycles,
                            )
                            plddt = float(
                                out["mean_plddt"]
                                .detach()
                                .float()
                                .cpu()
                                .reshape(-1)[0]
                            )
                            pdb_str: Optional[str] = None
                            if save_pdbs:
                                pdb_str = model.output_to_pdb(out)[0]
                        else:
                            pdb_str = model.infer_pdbs(
                                st,
                                num_recycles=num_recycles,
                            )[0]
                            plddt = calc_plddt_from_pdb_str(pdb_str)
                    if save_pdbs and pdb_str is not None:
                        pdb_path = out_dir / f"{label}_{entry_id}.pdb"
                        with open(pdb_path, "w", encoding="utf-8") as f:
                            f.write(pdb_str)
                    per_seq.append(
                        {
                            "entry_id": entry_id,
                            "seq_len": len(seq),
                            "plddt": plddt,
                        }
                    )
                    n_printed += 1
                    print(
                        f"[{label}] {n_printed}/{n_total} {entry_id} pLDDT={plddt:.2f}"
                    )
                except Exception as e2:
                    print(f"[{label}] {entry_id} failed: {e2}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_plddt = float(np.mean([x["plddt"] for x in per_seq])) if per_seq else float("nan")
    return {
        "label": label,
        "count_success": len(per_seq),
        "average_plddt": avg_plddt,
        "per_sequence": per_seq,
    }


def _default_batch_size() -> int:
    # ESMFold 显存大，默认 1 更不易 OOM；要提速可显式设 --batch-size 2+。
    return 1


def _limit_sequences(sequences: Dict[str, str], n: Optional[int]) -> Dict[str, str]:
    if n is None or n < 0:
        return sequences
    return dict(list(sequences.items())[:n])


def _partition_for_shard(
    sequences: Dict[str, str], shard_idx: int, shard_total: int
) -> Dict[str, str]:
    """第 shard_idx 份（0-based），共 shard_total 份，按 TSV 顺序均匀切分。"""
    if shard_total <= 0:
        return sequences
    items = list(sequences.items())
    return dict(
        [items[i] for i in range(len(items)) if i % shard_total == shard_idx]
    )


def _parse_gpus(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _merge_set_results(parts: List[dict], label: str) -> dict:
    all_per: List[dict] = []
    for p in parts:
        if p is None:
            continue
        all_per.extend(p.get("per_sequence", []))
    av = float(np.mean([x["plddt"] for x in all_per])) if all_per else float("nan")
    return {
        "label": label,
        "count_success": len(all_per),
        "average_plddt": av,
        "per_sequence": all_per,
    }


def load_local_esm2_without_regression(model_path: str):
    model_file = Path(model_path).resolve()
    if not model_file.name.endswith(".pt"):
        raise SystemExit("--esm2-ckpt 须指向以 .pt 结尾的主模型")
    if not model_file.exists():
        raise SystemExit(f"--esm2-ckpt 文件不存在: {model_file}")

    model_data = torch.load(str(model_file), map_location="cpu")
    return esm.pretrained.load_model_and_alphabet_core(
        model_file.stem,
        model_data,
        regression_data=None,
    )


def _needs_openfold_deprecated_key_convert(state_dict: dict) -> bool:
    """官方 fair-esm 的 esmfold_3B_v1.pt 用旧式 IPA 键名，当前 openfold 需多一层 .linear 。"""
    return any("ipa.linear_q_points.weight" in k for k in state_dict)


def load_esmfold_from_local_path(ckpt_path: str):
    """
    与 esm.esmfold.v1.pretrained._load_model 一致，但在加载前将旧版 OpenFold
    权重键名转为当前包中的命名（与 openfold.utils.import_weights.convert_deprecated_v1_keys 一致）。
    """
    from esm.esmfold.v1.esmfold import ESMFold
    from openfold.utils.import_weights import convert_deprecated_v1_keys

    model_path = Path(ckpt_path).resolve()
    if not str(model_path).endswith(".pt"):
        raise SystemExit("--esmfold-ckpt 须为以 .pt 结尾的本地文件路径")
    if not model_path.exists():
        raise SystemExit(f"--esmfold-ckpt 文件不存在: {model_path}")

    model_data = torch.load(str(model_path), map_location="cpu")
    model_state = model_data["model"]
    if _needs_openfold_deprecated_key_convert(model_state):
        model_state = convert_deprecated_v1_keys(model_state)
        model_data["model"] = model_state

    cfg = model_data["cfg"]["model"]
    model = ESMFold(esmfold_config=cfg)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())
    missing_essential_keys = [
        k
        for k in (expected_keys - found_keys)
        if not k.startswith("esm.")
    ]
    if missing_essential_keys:
        raise RuntimeError(
            "权重与当前 ESMFold 结构不匹配，缺键: "
            f"{', '.join(missing_essential_keys[:10])}"
            + (f" 等共{len(missing_essential_keys)}个" if len(missing_essential_keys) > 10 else "")
        )

    model.load_state_dict(model_state, strict=False)
    return model


def _build_worker_cmd(
    args: Any, shard_idx: int, shard_total: int, partial_path: Path
) -> List[str]:
    """仅用于多卡子进程，不传 --gpus，避免再 fork。"""
    script = Path(__file__).resolve()
    cmd: List[str] = [sys.executable, str(script)]
    cmd += [
        "--sequence-from-satoken-tsv",
        str(args.sequence_from_satoken_tsv),
        "--generated-sequence-tsv",
        str(args.generated_sequence_tsv),
        "--output-dir",
        str(args.output_dir),
        "--plddt-from",
        str(args.plddt_from),
    ]
    if args.chunk_size is not None:
        cmd += ["--chunk-size", str(args.chunk_size)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.save_pdbs:
        cmd.append("--save-pdbs")
    if args.num_recycles is not None:
        cmd += ["--num-recycles", str(args.num_recycles)]
    if args.max_sequences is not None:
        cmd += ["--max-sequences", str(args.max_sequences)]
    if args.esmfold_ckpt:
        cmd += ["--esmfold-ckpt", str(args.esmfold_ckpt)]
    if args.esm2_ckpt:
        cmd += ["--esm2-ckpt", str(args.esm2_ckpt)]
    cmd += [
        "--_shard-idx",
        str(shard_idx),
        "--_shard-total",
        str(shard_total),
        "--_partial-out",
        str(partial_path),
    ]
    return cmd


def _parent_run_multi_gpu(gpu_ids: List[int], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(gpu_ids)
    part_paths = [out_dir / ("_plddt_shard_%d.json" % i) for i in range(n)]
    procs: List[subprocess.Popen] = []
    for i, g in enumerate(gpu_ids):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        cmd = _build_worker_cmd(args, i, n, part_paths[i])
        procs.append(subprocess.Popen(cmd, env=env))
    for p in procs:
        rc = p.wait()
        if rc != 0:
            raise SystemExit("多卡子进程失败 returncode=%s" % rc)
    parts: List[dict] = []
    for pth in part_paths:
        with open(pth, "r", encoding="utf-8") as f:
            parts.append(json.load(f))
    r1p = [x["result_1"] for x in parts]
    r2p = [x["result_2"] for x in parts]
    result_1 = _merge_set_results(r1p, "sequence_from_satoken")
    result_2 = _merge_set_results(r2p, "generated_sequence")
    run_config: Dict[str, Any] = dict(parts[0].get("run_config", {}))
    run_config["multi_gpu"] = True
    run_config["gpus"] = ",".join(str(x) for x in gpu_ids)
    summary = {
        "sequence_from_satoken_average_plddt": result_1["average_plddt"],
        "generated_sequence_average_plddt": result_2["average_plddt"],
        "sequence_from_satoken_success_count": result_1["count_success"],
        "generated_sequence_success_count": result_2["count_success"],
    }
    with open(out_dir / "plddt_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "run_config": run_config,
                "sequence_from_satoken": result_1,
                "generated_sequence": result_2,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for pth in part_paths:
        try:
            pth.unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-from-satoken-tsv", required=True)
    parser.add_argument("--generated-sequence-tsv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="轴向注意力分块大小，更小的值（如 32、64）可显著省显存、略慢；"
        "OOM 时优先与 --batch-size 1 一起尝试。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每条前向同时预测的序列数，越大越快、显存越高；默认 1（易 OOM 勿随意加大）。",
    )
    parser.add_argument(
        "--plddt-from",
        choices=("network", "pdb_bfactor"),
        default="network",
        help="network：用 ESMFold 输出的 mean_plddT（快，与旧版 B-factor 均值可能略不同）；"
        "pdb_bfactor：写 PDB 后用 biotite 对 B-factor 取平均（与旧脚本一致，较慢）。",
    )
    parser.add_argument(
        "--save-pdbs",
        action="store_true",
        help="将预测结构写出为 PDB；不开启时可显著减少磁盘 I/O 与写文件时间。",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="ESMFold recycle 轮数，默认用模型内配置（多为 4）；"
        "改为 1~2 可明显加速、结构质量可能下降。",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="仅处理每条 TSV 前 N 条，用于试跑/调参。",
    )
    parser.add_argument(
        "--esmfold-ckpt",
        type=str,
        default=None,
        help=(
            "本地上传后的 ESMFold 结构权重，避免从网络下载；路径须以 .pt 结尾。官方: "
            "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt"
        ),
    )
    parser.add_argument(
        "--esm2-ckpt",
        type=str,
        default=None,
        help=(
            "本地上传后的 ESM-2 (3B) 主模型 .pt，供 ESMFold 内部 esm2_t36_3B_UR50D 使用，避免再下一遍。"
            "这里会忽略 contact regression 权重，因为本脚本只做折叠与 pLDDT。主模型: "
            "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt"
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "多卡数据并行。逗号分隔的 GPU 编号，例如 0,1,2,3 会启动 4 个子进程、"
            "每进程绑定一张卡并处理约 1/4 的序列。单进程仍可用环境变量 "
            "CUDA_VISIBLE_DEVICES=2 等指定单卡；勿与此参数混用多进程时重复设。"
        ),
    )
    parser.add_argument("--_shard-idx", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_shard-total", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_partial-out", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.gpus and args._shard_idx is None and len(_parse_gpus(args.gpus)) > 1:
        _parent_run_multi_gpu(_parse_gpus(args.gpus), args)
        return

    if (
        args._shard_idx is not None
        or args._shard_total is not None
        or args._partial_out is not None
    ):
        if (
            args._shard_idx is None
            or args._shard_total is None
            or not args._partial_out
        ):
            raise SystemExit(
                "内部错误：子进程分片需同时提供 --_shard-idx、--_shard-total 与 --_partial-out"
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda, "matmul"
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    # ESMFold 构造时会拉取 ESM-2 3B；本机已有 .pt 时临时替换为本地加载，并忽略 contact regression
    esm2_restore = None
    if args.esm2_ckpt:
        esm2_path = str(Path(args.esm2_ckpt).resolve())
        esm2_restore = esm.pretrained.esm2_t36_3B_UR50D
        esm.pretrained.esm2_t36_3B_UR50D = lambda: load_local_esm2_without_regression(
            esm2_path
        )
    try:
        if args.esmfold_ckpt:
            model = load_esmfold_from_local_path(args.esmfold_ckpt)
        else:
            model = esm.pretrained.esmfold_v1()
    finally:
        if esm2_restore is not None:
            esm.pretrained.esm2_t36_3B_UR50D = esm2_restore
    model = model.eval().to(device)

    if args.chunk_size is not None:
        model.set_chunk_size(args.chunk_size)

    bs = args.batch_size if args.batch_size is not None else _default_batch_size()

    seq_from_satoken = _limit_sequences(
        read_sequences_from_tsv(
            args.sequence_from_satoken_tsv, "sequence_from_satoken"
        ),
        args.max_sequences,
    )
    generated_sequence = _limit_sequences(
        read_sequences_from_tsv(
            args.generated_sequence_tsv, "generated_sequence"
        ),
        args.max_sequences,
    )

    if args._shard_idx is not None and args._shard_total is not None:
        seq_from_satoken = _partition_for_shard(
            seq_from_satoken, args._shard_idx, args._shard_total
        )
        generated_sequence = _partition_for_shard(
            generated_sequence, args._shard_idx, args._shard_total
        )

    run_config: Dict[str, Any] = {
        "device": device,
        "batch_size": bs,
        "plddt_from": args.plddt_from,
        "save_pdbs": args.save_pdbs,
        "chunk_size": args.chunk_size,
        "num_recycles": args.num_recycles,
        "max_sequences": args.max_sequences,
    }
    if args._shard_idx is not None and args._shard_total is not None:
        run_config["shard"] = "%d/%d" % (args._shard_idx, args._shard_total)

    result_1 = run_one_set(
        model,
        seq_from_satoken,
        out_dir / "sequence_from_satoken_pdbs",
        "sequence_from_satoken",
        batch_size=bs,
        plddt_from=args.plddt_from,
        save_pdbs=args.save_pdbs,
        num_recycles=args.num_recycles,
    )
    result_2 = run_one_set(
        model,
        generated_sequence,
        out_dir / "generated_sequence_pdbs",
        "generated_sequence",
        batch_size=bs,
        plddt_from=args.plddt_from,
        save_pdbs=args.save_pdbs,
        num_recycles=args.num_recycles,
    )

    summary = {
        "sequence_from_satoken_average_plddt": result_1["average_plddt"],
        "generated_sequence_average_plddt": result_2["average_plddt"],
        "sequence_from_satoken_success_count": result_1["count_success"],
        "generated_sequence_success_count": result_2["count_success"],
    }

    if args._partial_out:
        with open(args._partial_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "run_config": run_config,
                    "result_1": result_1,
                    "result_2": result_2,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    with open(out_dir / "plddt_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "run_config": run_config,
                "sequence_from_satoken": result_1,
                "generated_sequence": result_2,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()