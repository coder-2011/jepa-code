from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.dataset_helpers import build_eval_dataloader, list_split_shards
from scripts.inspect_checkpoint import effective_rank, load_checkpoint, resolve_parameter_golf_assets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare activation statistics between two checkpoints")
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--parameter-golf-root", type=Path, required=True)
    parser.add_argument("--variant", default="sp1024")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    assert args.batch_size > 0, "batch_size must be positive"
    assert args.eval_batches > 0, "eval_batches must be positive"
    return args


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean()


def masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].std(unbiased=False)


@torch.no_grad()
def collect_stats(model, loader, *, device: torch.device) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "num_batches": 0,
        "num_tokens": 0,
        "loss": 0.0,
        "loss_lm": 0.0,
        "loss_jepa": 0.0,
        "loss_sigreg": 0.0,
        "final_state_mean_abs": 0.0,
        "final_state_std": 0.0,
        "final_state_abs_max": 0.0,
        "tail_delta_norm": [0.0 for _ in model.blocks],
        "valid_delta_norm": [0.0 for _ in model.blocks],
        "tail_target_delta_norm": [0.0 for _ in model.blocks],
        "valid_target_delta_norm": [0.0 for _ in model.blocks],
        "z_std_mean": [0.0 for _ in model.blocks],
        "z_std_min": [0.0 for _ in model.blocks],
        "z_effective_rank": [0.0 for _ in model.blocks],
        "z_abs_max": [0.0 for _ in model.blocks],
    }

    for batch in loader:
        input_ids, labels = (tensor.to(device) for tensor in batch)
        outputs = model(input_ids=input_ids, labels=labels)
        jepa_valid_mask = outputs["jepa_valid_mask"].to(torch.bool)
        tail_mask = ~jepa_valid_mask
        final_states = outputs["final_states"].detach().float()
        totals["num_batches"] += 1
        totals["num_tokens"] += int(input_ids.numel())
        totals["loss"] += float(outputs["loss"])
        totals["loss_lm"] += float(outputs["loss_main"])
        totals["loss_jepa"] += float(outputs["loss_jepa"])
        totals["loss_sigreg"] += float(outputs["loss_sigreg"])
        totals["final_state_mean_abs"] += float(final_states.abs().mean())
        totals["final_state_std"] += float(final_states.std(unbiased=False))
        totals["final_state_abs_max"] += float(final_states.abs().max())

        for index, z in enumerate(outputs["z"]):
            z = z.detach().float()
            delta = outputs["deltas"][index].detach().float()
            target_delta = outputs["targets"][index].detach().float() - z
            z_flat = z.reshape(-1, z.shape[-1])
            z_std = z_flat.std(dim=0, unbiased=False)
            totals["z_std_mean"][index] += float(z_std.mean())
            totals["z_std_min"][index] += float(z_std.min())
            totals["z_effective_rank"][index] += effective_rank(z)
            totals["z_abs_max"][index] += float(z.abs().max())
            totals["tail_delta_norm"][index] += float(delta[tail_mask].norm(dim=-1).mean())
            totals["valid_delta_norm"][index] += float(delta[jepa_valid_mask].norm(dim=-1).mean())
            totals["tail_target_delta_norm"][index] += float(target_delta[tail_mask].norm(dim=-1).mean())
            totals["valid_target_delta_norm"][index] += float(target_delta[jepa_valid_mask].norm(dim=-1).mean())

    count = float(totals["num_batches"])
    assert count > 0, "eval loader produced no batches"
    for key, value in list(totals.items()):
        if isinstance(value, list):
            totals[key] = [inner / count for inner in value]
        elif key not in {"num_batches", "num_tokens"}:
            totals[key] = value / count
    return totals


def compare_stats(stats_a: dict[str, Any], stats_b: dict[str, Any]) -> dict[str, Any]:
    layer_count = len(stats_a["z_std_mean"])
    summary = {
        "checkpoint_a": stats_a,
        "checkpoint_b": stats_b,
        "drift": {
            "loss": stats_b["loss"] - stats_a["loss"],
            "loss_lm": stats_b["loss_lm"] - stats_a["loss_lm"],
            "loss_jepa": stats_b["loss_jepa"] - stats_a["loss_jepa"],
            "loss_sigreg": stats_b["loss_sigreg"] - stats_a["loss_sigreg"],
            "final_state_mean_abs": stats_b["final_state_mean_abs"] - stats_a["final_state_mean_abs"],
            "final_state_std": stats_b["final_state_std"] - stats_a["final_state_std"],
            "final_state_abs_max": stats_b["final_state_abs_max"] - stats_a["final_state_abs_max"],
            "layers": [],
        },
        "warnings": [],
    }
    for index in range(layer_count):
        layer = {
            "layer": index,
            "z_std_mean": stats_b["z_std_mean"][index] - stats_a["z_std_mean"][index],
            "z_std_min": stats_b["z_std_min"][index] - stats_a["z_std_min"][index],
            "z_effective_rank": stats_b["z_effective_rank"][index] - stats_a["z_effective_rank"][index],
            "z_abs_max": stats_b["z_abs_max"][index] - stats_a["z_abs_max"][index],
            "valid_delta_norm": stats_b["valid_delta_norm"][index] - stats_a["valid_delta_norm"][index],
            "tail_delta_norm": stats_b["tail_delta_norm"][index] - stats_a["tail_delta_norm"][index],
            "tail_to_valid_delta_ratio_a": stats_a["tail_delta_norm"][index] / max(stats_a["valid_delta_norm"][index], 1e-12),
            "tail_to_valid_delta_ratio_b": stats_b["tail_delta_norm"][index] / max(stats_b["valid_delta_norm"][index], 1e-12),
            "tail_to_valid_target_ratio_a": stats_a["tail_target_delta_norm"][index] / max(stats_a["valid_target_delta_norm"][index], 1e-12),
            "tail_to_valid_target_ratio_b": stats_b["tail_target_delta_norm"][index] / max(stats_b["valid_target_delta_norm"][index], 1e-12),
        }
        summary["drift"]["layers"].append(layer)
        if stats_b["z_std_min"][index] < 1e-4:
            summary["warnings"].append(f"layer {index} z_std_min collapsed by checkpoint B")
        if layer["tail_to_valid_delta_ratio_b"] > 1.25:
            summary["warnings"].append(
                f"layer {index} tail delta norm exceeds valid-position delta norm by >25% at checkpoint B"
            )
        if layer["tail_to_valid_target_ratio_b"] > 1.25:
            summary["warnings"].append(
                f"layer {index} tail target-delta norm exceeds valid-position target-delta norm by >25% at checkpoint B"
            )
    if stats_b["final_state_abs_max"] > stats_a["final_state_abs_max"] * 1.5:
        summary["warnings"].append("final-state abs max grew by more than 50% between checkpoints")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    assets = resolve_parameter_golf_assets(args.parameter_golf_root, args.variant)

    model_a, checkpoint_a, _ = load_checkpoint(args.checkpoint_a, device)
    model_b, checkpoint_b, _ = load_checkpoint(args.checkpoint_b, device)
    seq_len = model_a.config.max_length if args.seq_len is None else args.seq_len
    eval_loader = build_eval_dataloader(
        list_split_shards(assets.dataset_root, "val"),
        batch_size=args.batch_size,
        seq_len=seq_len,
        max_batches=args.eval_batches,
        pin_memory=device.type == "cuda",
    )

    stats_a = collect_stats(model_a, eval_loader, device=device)
    eval_loader = build_eval_dataloader(
        list_split_shards(assets.dataset_root, "val"),
        batch_size=args.batch_size,
        seq_len=seq_len,
        max_batches=args.eval_batches,
        pin_memory=device.type == "cuda",
    )
    stats_b = collect_stats(model_b, eval_loader, device=device)
    summary = compare_stats(stats_a, stats_b)
    summary["checkpoint_a"]["step"] = checkpoint_a.get("step")
    summary["checkpoint_b"]["step"] = checkpoint_b.get("step")

    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.out is not None:
        args.out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
