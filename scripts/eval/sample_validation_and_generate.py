from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any

import torch
import tree
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import GenerationConfig

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from utils.config_utils import load_config_from_yaml
from utils.init_utils import construct_class_by_name
from utils.sequence_utils import extract_amino_acid_sequence_from_satoken


class _Logger:
    def info(self, msg: str) -> None:
        print(msg, file=sys.stderr)


@dataclass
class ModelArtifacts:
    model: torch.nn.Module
    dataloader: DataLoader


def _resolve_checkpoint_file(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "pytorch_model.bin")
        if os.path.isfile(candidate):
            return candidate
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"checkpoint 不存在: {path}")


def _get_dataset_entry_ids(dataset) -> list[str]:
    if hasattr(dataset, "samples"):
        return [sample["entry_id"] for sample in dataset.samples]
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return [dataset.dataset.samples[idx]["entry_id"] for idx in dataset.indices]
    return [dataset[idx]["entry_id"] for idx in range(len(dataset))]


def _build_generation_loader(
    config_path: str,
    checkpoint_path: str,
    sampled_ids: list[str],
    batch_size: int,
    device: torch.device,
) -> ModelArtifacts:
    cfg = load_config_from_yaml(config_path)
    logger = _Logger()

    model = construct_class_by_name(**cfg.Model.kwargs.to_dict(), logger=logger)
    state_dict = torch.load(_resolve_checkpoint_file(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    datamodule = construct_class_by_name(
        **cfg.Datamodule.kwargs.to_dict(),
        logger=logger,
        tokenizer=model.tokenizer,
        condition_tokenizer=getattr(model, "condition_tokenizer", None),
    )
    datamodule.set_val_dataset()
    dataset = datamodule.val_dataset
    dataset_entry_ids = _get_dataset_entry_ids(dataset)
    id_to_idx = {entry_id: idx for idx, entry_id in enumerate(dataset_entry_ids)}
    missing_ids = [entry_id for entry_id in sampled_ids if entry_id not in id_to_idx]
    if missing_ids:
        raise KeyError(f"以下 sampled entry_id 不在验证集里: {missing_ids[:5]}")

    subset = Subset(dataset, [id_to_idx[entry_id] for entry_id in sampled_ids])
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.Datamodule.kwargs.num_workers_per_gpu,
        pin_memory=True,
        collate_fn=dataset.collate,
    )
    return ModelArtifacts(model=model, dataloader=loader)


def _load_raw_validation_sets(
    satoken_config_path: str, sequence_config_path: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    sat_cfg = load_config_from_yaml(satoken_config_path)
    seq_cfg = load_config_from_yaml(sequence_config_path)
    sat_val = torch.load(sat_cfg.Datamodule.kwargs.val_split_path, map_location="cpu")
    seq_val = torch.load(seq_cfg.Datamodule.kwargs.val_split_path, map_location="cpu")
    return sat_val, seq_val


def _sample_common_entry_ids(
    sat_val: dict[str, Any], seq_val: dict[str, Any], sample_size: int, seed: int
) -> list[str]:
    common_ids = sorted(set(sat_val.keys()) & set(seq_val.keys()))
    if sample_size > len(common_ids):
        raise ValueError(
            f"sample_size={sample_size} 超过共同验证样本数 {len(common_ids)}"
        )
    rng = random.Random(seed)
    return rng.sample(common_ids, sample_size)


@torch.inference_mode()
def _generate_sequences(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    generation_config: GenerationConfig,
    desc: str,
) -> dict[str, str]:
    generated_by_id: dict[str, str] = {}
    for batch in tqdm(dataloader, desc=desc):
        batch = tree.map_structure(
            lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
            batch,
        )
        res = model.generate(**batch, generation_config=generation_config)
        for entry_id, generated_seq in zip(batch["entry_id"], res["output_seqs"]):
            generated_by_id[str(entry_id)] = generated_seq
    return generated_by_id


def _metadata_for_entry(entry_id: str, sat_val: dict[str, Any], seq_val: dict[str, Any]) -> dict[str, str]:
    substrate, product, gt_satoken = sat_val[entry_id][0], sat_val[entry_id][1], sat_val[entry_id][-1]
    gt_sequence = seq_val[entry_id][-1]
    reaction = f"{substrate}>>{product}"
    return {
        "entry_id": entry_id,
        "reaction_smiles": reaction,
        "raw_reaction": reaction,
        "gt_satoken": gt_satoken,
        "gt_sequence": gt_sequence,
    }


def _write_tsv(path: str, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _default_output_dir(
    satoken_ckpt: str, sequence_ckpt: str, sample_size: int, seed: int
) -> str:
    sat_name = os.path.basename(os.path.normpath(satoken_ckpt))
    seq_name = os.path.basename(os.path.normpath(sequence_ckpt))
    return os.path.join(
        _ROOT,
        "output",
        "validation_sample_generations",
        f"{sat_name}__vs__{seq_name}_n{sample_size}_seed{seed}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从验证集中抽样若干样本，分别生成 satoken / sequence，并保存三份结果文件。"
    )
    parser.add_argument("--satoken-config", required=True)
    parser.add_argument("--satoken-checkpoint", required=True)
    parser.add_argument("--sequence-config", required=True)
    parser.add_argument("--sequence-checkpoint", required=True)
    parser.add_argument(
        "--generation-config",
        default=os.path.join(_ROOT, "configs", "generation", "argmax.yaml"),
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    generation_cfg = load_config_from_yaml(args.generation_config)
    generation_config = GenerationConfig(**generation_cfg.GenerationConfig.to_dict())

    sat_val, seq_val = _load_raw_validation_sets(
        args.satoken_config, args.sequence_config
    )
    sampled_ids = _sample_common_entry_ids(
        sat_val=sat_val, seq_val=seq_val, sample_size=args.sample_size, seed=args.seed
    )

    output_dir = args.output_dir or _default_output_dir(
        args.satoken_checkpoint,
        args.sequence_checkpoint,
        args.sample_size,
        args.seed,
    )
    os.makedirs(output_dir, exist_ok=True)

    satoken_artifacts = _build_generation_loader(
        config_path=args.satoken_config,
        checkpoint_path=args.satoken_checkpoint,
        sampled_ids=sampled_ids,
        batch_size=args.batch_size,
        device=device,
    )
    satoken_generated = _generate_sequences(
        model=satoken_artifacts.model,
        dataloader=satoken_artifacts.dataloader,
        device=device,
        generation_config=generation_config,
        desc="Generating satoken",
    )

    sequence_artifacts = _build_generation_loader(
        config_path=args.sequence_config,
        checkpoint_path=args.sequence_checkpoint,
        sampled_ids=sampled_ids,
        batch_size=args.batch_size,
        device=device,
    )
    sequence_generated = _generate_sequences(
        model=sequence_artifacts.model,
        dataloader=sequence_artifacts.dataloader,
        device=device,
        generation_config=generation_config,
        desc="Generating sequence",
    )

    direct_satoken_rows = []
    direct_sequence_rows = []
    sequence_from_satoken_rows = []
    for entry_id in sampled_ids:
        meta = _metadata_for_entry(entry_id, sat_val=sat_val, seq_val=seq_val)
        generated_satoken = satoken_generated[entry_id]
        generated_sequence = sequence_generated[entry_id]
        extracted_sequence = extract_amino_acid_sequence_from_satoken(generated_satoken)

        direct_satoken_rows.append(
            [
                meta["entry_id"],
                meta["reaction_smiles"],
                meta["raw_reaction"],
                meta["gt_satoken"],
                generated_satoken,
            ]
        )
        direct_sequence_rows.append(
            [
                meta["entry_id"],
                meta["reaction_smiles"],
                meta["raw_reaction"],
                meta["gt_sequence"],
                generated_sequence,
            ]
        )
        sequence_from_satoken_rows.append(
            [
                meta["entry_id"],
                meta["reaction_smiles"],
                meta["raw_reaction"],
                meta["gt_satoken"],
                meta["gt_sequence"],
                generated_satoken,
                extracted_sequence,
            ]
        )

    direct_satoken_path = os.path.join(output_dir, "direct_satoken_generation.tsv")
    direct_sequence_path = os.path.join(output_dir, "direct_sequence_generation.tsv")
    sequence_from_satoken_path = os.path.join(output_dir, "sequence_from_satoken.tsv")
    metadata_path = os.path.join(output_dir, "metadata.json")

    _write_tsv(
        direct_satoken_path,
        ["entry_id", "reaction_smiles", "raw_reaction", "gt_satoken", "generated_satoken"],
        direct_satoken_rows,
    )
    _write_tsv(
        direct_sequence_path,
        ["entry_id", "reaction_smiles", "raw_reaction", "gt_sequence", "generated_sequence"],
        direct_sequence_rows,
    )
    _write_tsv(
        sequence_from_satoken_path,
        [
            "entry_id",
            "reaction_smiles",
            "raw_reaction",
            "gt_satoken",
            "gt_sequence",
            "generated_satoken",
            "sequence_from_satoken",
        ],
        sequence_from_satoken_rows,
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_size": args.sample_size,
                "seed": args.seed,
                "device": str(device),
                "generation_config": os.path.abspath(args.generation_config),
                "satoken_config": os.path.abspath(args.satoken_config),
                "satoken_checkpoint": os.path.abspath(args.satoken_checkpoint),
                "sequence_config": os.path.abspath(args.sequence_config),
                "sequence_checkpoint": os.path.abspath(args.sequence_checkpoint),
                "sampled_entry_ids": sampled_ids,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(
            {
                "output_dir": output_dir,
                "direct_satoken_generation": direct_satoken_path,
                "direct_sequence_generation": direct_sequence_path,
                "sequence_from_satoken": sequence_from_satoken_path,
                "metadata": metadata_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
