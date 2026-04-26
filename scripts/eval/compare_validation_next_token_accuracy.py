from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import tree
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from utils.config_utils import load_config_from_yaml
from utils.init_utils import construct_class_by_name


class _Logger:
    def info(self, msg: str) -> None:
        print(msg, file=sys.stderr)


@dataclass
class EvalArtifacts:
    model: torch.nn.Module
    val_loader: torch.utils.data.DataLoader
    tokenizer: Any


def _resolve_checkpoint_file(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "pytorch_model.bin")
        if os.path.isfile(candidate):
            return candidate
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"checkpoint 不存在: {path}")


def _build_eval_artifacts(
    config_path: str,
    checkpoint_path: str,
    batch_size: int | None,
    device: torch.device,
) -> EvalArtifacts:
    cfg = load_config_from_yaml(config_path)
    logger = _Logger()

    model = construct_class_by_name(**cfg.Model.kwargs.to_dict(), logger=logger)
    state_dict = torch.load(_resolve_checkpoint_file(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    datamodule_kwargs = cfg.Datamodule.kwargs.to_dict()
    if batch_size is not None:
        datamodule_kwargs["eval_batch_size"] = int(batch_size)
    datamodule = construct_class_by_name(
        **datamodule_kwargs,
        logger=logger,
        tokenizer=model.tokenizer,
        condition_tokenizer=getattr(model, "condition_tokenizer", None),
    )
    datamodule.set_val_dataset()
    val_loader = datamodule.val_dataloader()
    return EvalArtifacts(model=model, val_loader=val_loader, tokenizer=model.tokenizer)


def _make_satoken_equivalence_map(tokenizer: Any) -> torch.Tensor:
    vocab_size = len(tokenizer)
    token_to_class: dict[tuple[str, str], int] = {}
    class_ids: list[int] = []

    def _class_for_token(token: str) -> int:
        if token.startswith("<") and token.endswith(">"):
            key = ("special", token)
        else:
            uppercase = [ch for ch in token if ch.isupper()]
            if uppercase:
                key = ("aa", uppercase[0])
            else:
                key = ("token", token)
        if key not in token_to_class:
            token_to_class[key] = len(token_to_class)
        return token_to_class[key]

    for token_id in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id)
        class_ids.append(_class_for_token(token))

    return torch.tensor(class_ids, dtype=torch.long)


def _forward_for_next_token_eval(
    model: torch.nn.Module,
    batch: dict[str, Any],
    autocast_ctx,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_ctx:
        reaction_tokens, reaction_mask = model.forward_encoder(
            substrate_atom_tokens=batch["substrate_atom_tokens"],
            substrate_atom_masks=batch["substrate_atom_masks"],
            product_atom_tokens=batch["product_atom_tokens"],
            product_atom_masks=batch["product_atom_masks"],
        )
        inputs_embeds, attention_mask, labels = model._build_decoder_inputs(
            seq_ids=batch["seq_ids"],
            seq_masks=batch["seq_masks"],
            reaction_tokens=reaction_tokens,
            reaction_mask=reaction_mask,
        )
        logits = model.qwen3(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits

    pred_ids = logits[..., :-1, :].argmax(dim=-1)
    target_ids = labels[..., 1:]
    return pred_ids, target_ids


@torch.inference_mode()
def _evaluate_sequence_exact(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    total_tokens = 0
    correct_tokens = 0
    total_samples = 0
    use_autocast = device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_autocast
        else nullcontext()
    )

    for batch_idx, batch in tqdm(
        enumerate(val_loader),
        total=len(val_loader) if max_batches is None else min(len(val_loader), max_batches),
        desc="Evaluating sequence",
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = tree.map_structure(
            lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
            batch,
        )
        pred_ids, target_ids = _forward_for_next_token_eval(model, batch, autocast_ctx)
        valid_mask = target_ids.ne(-100)
        total_tokens += int(valid_mask.sum().item())
        correct_tokens += int(((pred_ids == target_ids) & valid_mask).sum().item())
        total_samples += int(batch["seq_ids"].shape[0])

    return {
        "mode": "sequence_exact",
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "correct_tokens": correct_tokens,
        "accuracy": (correct_tokens / total_tokens) if total_tokens else 0.0,
    }


@torch.inference_mode()
def _evaluate_satoken_fair(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    tokenizer: Any,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    eq_map = _make_satoken_equivalence_map(tokenizer).to(device)
    total_tokens = 0
    strict_correct_tokens = 0
    fair_correct_tokens = 0
    total_samples = 0
    use_autocast = device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_autocast
        else nullcontext()
    )

    for batch_idx, batch in tqdm(
        enumerate(val_loader),
        total=len(val_loader) if max_batches is None else min(len(val_loader), max_batches),
        desc="Evaluating satoken",
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = tree.map_structure(
            lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
            batch,
        )
        pred_ids, target_ids = _forward_for_next_token_eval(model, batch, autocast_ctx)
        valid_mask = target_ids.ne(-100)
        safe_target_ids = target_ids.masked_fill(~valid_mask, 0)

        total_tokens += int(valid_mask.sum().item())
        strict_correct_tokens += int(((pred_ids == target_ids) & valid_mask).sum().item())
        fair_correct_tokens += int(
            ((eq_map[pred_ids] == eq_map[safe_target_ids]) & valid_mask).sum().item()
        )
        total_samples += int(batch["seq_ids"].shape[0])

    strict_accuracy = strict_correct_tokens / total_tokens if total_tokens else 0.0
    fair_accuracy = fair_correct_tokens / total_tokens if total_tokens else 0.0
    return {
        "mode": "satoken_aa_match",
        "rule": "内容 token 只要大写氨基酸字母一致即记为正确；特殊 token 仍需完全一致。",
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "strict_correct_tokens": strict_correct_tokens,
        "fair_correct_tokens": fair_correct_tokens,
        "strict_accuracy": strict_accuracy,
        "fair_accuracy": fair_accuracy,
    }


def _default_output_path(satoken_ckpt: str, sequence_ckpt: str) -> str:
    sat_name = os.path.basename(os.path.normpath(satoken_ckpt))
    seq_name = os.path.basename(os.path.normpath(sequence_ckpt))
    return os.path.join(
        _ROOT,
        "output",
        "validation_accuracy_compare",
        f"{sat_name}__vs__{seq_name}.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较 satoken 与 sequence 模型在验证集上的 next-token prediction accuracy。"
    )
    parser.add_argument("--satoken-config", required=True)
    parser.add_argument("--satoken-checkpoint", required=True)
    parser.add_argument("--sequence-config", required=True)
    parser.add_argument("--sequence-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_json = args.output_json or _default_output_path(
        args.satoken_checkpoint, args.sequence_checkpoint
    )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    satoken_artifacts = _build_eval_artifacts(
        config_path=args.satoken_config,
        checkpoint_path=args.satoken_checkpoint,
        batch_size=args.batch_size,
        device=device,
    )
    satoken_metrics = _evaluate_satoken_fair(
        model=satoken_artifacts.model,
        val_loader=satoken_artifacts.val_loader,
        tokenizer=satoken_artifacts.tokenizer,
        device=device,
        max_batches=args.max_batches,
    )

    sequence_artifacts = _build_eval_artifacts(
        config_path=args.sequence_config,
        checkpoint_path=args.sequence_checkpoint,
        batch_size=args.batch_size,
        device=device,
    )
    sequence_metrics = _evaluate_sequence_exact(
        model=sequence_artifacts.model,
        val_loader=sequence_artifacts.val_loader,
        device=device,
        max_batches=args.max_batches,
    )

    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "satoken": {
            "config": os.path.abspath(args.satoken_config),
            "checkpoint": os.path.abspath(args.satoken_checkpoint),
            **satoken_metrics,
        },
        "sequence": {
            "config": os.path.abspath(args.sequence_config),
            "checkpoint": os.path.abspath(args.sequence_checkpoint),
            **sequence_metrics,
        },
        "comparison": {
            "satoken_fair_minus_sequence": satoken_metrics["fair_accuracy"]
            - sequence_metrics["accuracy"],
            "satoken_strict_minus_sequence": satoken_metrics["strict_accuracy"]
            - sequence_metrics["accuracy"],
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n结果已写入: {output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
