from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.dataset_helpers import build_eval_dataloader, list_split_shards
from scripts.inspect_checkpoint import load_checkpoint, resolve_parameter_golf_assets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe LM behavior and representation geometry from a checkpoint")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True, help="Checkpoint path; repeatable")
    parser.add_argument("--parameter-golf-root", type=Path, required=True)
    parser.add_argument("--variant", default="sp1024")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=8)
    args = parser.parse_args(argv)
    assert args.batch_size > 0, "batch_size must be positive"
    assert args.seq_len > 1, "seq_len must be at least 2"
    assert args.eval_batches > 0, "eval_batches must be positive"
    return args


def mean_pairwise_cosine(x: torch.Tensor) -> float:
    assert x.ndim == 3, "expected (B, L, D) tensor"
    y = F.normalize(x.float().reshape(-1, x.shape[-1]), dim=-1)
    gram = y @ y.T
    mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    return float(gram[mask].mean())


def mean_logit_correlation(logits: torch.Tensor) -> float:
    flat = logits.float().reshape(-1, logits.shape[-1])
    centered = flat - flat.mean(dim=-1, keepdim=True)
    standardized = centered / (centered.std(dim=-1, keepdim=True, unbiased=False) + 1e-8)
    corr = (standardized @ standardized.T) / standardized.shape[-1]
    mask = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
    return float(corr[mask].mean())


def top_pc_fraction(x: torch.Tensor) -> float:
    flat = x.float().reshape(-1, x.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.cpu())
    spectrum = singular_values.square()
    return float(spectrum[0] / spectrum.sum().clamp_min(1e-12))


@torch.no_grad()
def probe_checkpoint(
    checkpoint_path: Path,
    *,
    parameter_golf_root: Path,
    variant: str,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
) -> dict[str, Any]:
    assets = resolve_parameter_golf_assets(parameter_golf_root, variant)
    loader = build_eval_dataloader(
        list_split_shards(assets.dataset_root, "val"),
        batch_size=batch_size,
        seq_len=seq_len,
        max_batches=eval_batches,
        pin_memory=device.type == "cuda",
    )
    model, checkpoint, _ = load_checkpoint(checkpoint_path, device)
    step = int(checkpoint.get("step", 0))

    total_tokens = 0
    total_correct = 0
    total_nll = 0.0
    total_entropy = 0.0
    total_top1p = 0.0
    total_final_cos = 0.0
    total_logit_corr = 0.0
    total_final_top_pc = 0.0
    total_logit_top_pc = 0.0
    batches = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels, step=step)
        logits = outputs["logits"]
        probs = logits.float().softmax(dim=-1)
        predictions = logits.argmax(dim=-1)
        total_correct += int((predictions == labels).sum())
        total_tokens += labels.numel()
        total_nll += float(
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        )
        total_entropy += float((-(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)).mean())
        total_top1p += float(probs.max(dim=-1).values.mean())
        total_final_cos += mean_pairwise_cosine(outputs["final_states"])
        total_logit_corr += mean_logit_correlation(logits)
        total_final_top_pc += top_pc_fraction(outputs["final_states"])
        total_logit_top_pc += top_pc_fraction(logits)
        batches += 1

    assert batches > 0, "eval loader produced no batches"
    avg_nll = total_nll / total_tokens
    return {
        "checkpoint": str(checkpoint_path),
        "step": step,
        "tokens": total_tokens,
        "acc1": total_correct / total_tokens,
        "nll": avg_nll,
        "bits_per_token": avg_nll / math.log(2.0),
        "entropy": total_entropy / batches,
        "top1p": total_top1p / batches,
        "final_state_pairwise_cosine": total_final_cos / batches,
        "logit_mean_correlation": total_logit_corr / batches,
        "final_state_top_pc_fraction": total_final_top_pc / batches,
        "logit_top_pc_fraction": total_logit_top_pc / batches,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    results = [
        probe_checkpoint(
            checkpoint_path,
            parameter_golf_root=args.parameter_golf_root,
            variant=args.variant,
            device=device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_batches=args.eval_batches,
        )
        for checkpoint_path in args.checkpoint
    ]
    print(json.dumps(results[0] if len(results) == 1 else results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
