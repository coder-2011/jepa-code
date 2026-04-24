from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class RunProfile:
    batch_size: int
    seq_len: int
    train_shards: int
    max_steps: int
    log_every: int
    eval_every: int
    eval_batches: int


PROFILES = {
    "smoke": RunProfile(2, 64, 1, 2, 1, 1, 2),
    "full": RunProfile(8, 128, 1, 5000, 250, 2500, 8),
}


RESULTS_COLUMNS = [
    "timestamp_utc",
    "commit",
    "run_name",
    "profile",
    "status",
    "description",
    "val_bpb",
    "eval_loss",
    "eval_loss_lm",
    "eval_loss_jepa",
    "eval_loss_sigreg",
    "train_loss",
    "train_loss_lm",
    "train_loss_jepa",
    "train_loss_sigreg",
    "lambda_jepa",
    "beta_sigreg",
    "jepa_aux_dropped",
    "jepa_dropout_fraction",
    "peak_memory_gb",
    "tokens_per_sec",
    "tokens_processed",
    "num_steps",
    "wall_seconds",
    "device",
    "dtype",
    "optimizer",
    "lr",
    "weight_decay",
    "batch_size",
    "seq_len",
    "train_shards",
    "eval_batches",
    "config",
    "checkpoint",
    "run_dir",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoresearch wrapper around scripts.train_intertwined_hjepa")
    parser.add_argument("--config", type=Path, default=ROOT / "intertwined_hjepa.yaml")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--parameter-golf-root", type=Path, required=True)
    parser.add_argument("--variant", default="sp1024")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--profile", default="full", choices=sorted(PROFILES))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--train-shards", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw8bit", "adamw4bit", "adamwfp8"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "autoresearch")
    parser.add_argument("--results-path", type=Path, default=Path(__file__).resolve().parent / "results.tsv")
    parser.add_argument("--description", default="baseline")
    parser.add_argument("--status", default="pending", choices=["baseline", "keep", "discard", "crash", "pending"])
    parser.add_argument("--wandb-project", default="intertwined-hjepa")
    parser.add_argument("--wandb-mode", default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument("--torchao-float8", action="store_true")
    parser.add_argument("--torchao-float8-recipe", default="tensorwise", choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"])
    parser.add_argument("--jepa-dropout-rate", type=float, default=None)
    parser.add_argument("--auxiliary-layer-start", type=int, default=None)
    parser.add_argument("--auxiliary-layer-stride", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="Compile training on non-CUDA devices; CUDA always compiles")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--write-results", dest="write_results", action="store_true")
    parser.add_argument("--no-write-results", dest="write_results", action="store_false")
    parser.set_defaults(write_results=True)
    return parser.parse_args(argv)


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    profile = PROFILES[args.profile]
    args.batch_size = profile.batch_size if args.batch_size is None else args.batch_size
    args.seq_len = profile.seq_len if args.seq_len is None else args.seq_len
    args.train_shards = profile.train_shards if args.train_shards is None else args.train_shards
    args.max_steps = profile.max_steps if args.max_steps is None else args.max_steps
    args.log_every = profile.log_every if args.log_every is None else args.log_every
    args.eval_every = profile.eval_every if args.eval_every is None else args.eval_every
    args.eval_batches = profile.eval_batches if args.eval_batches is None else args.eval_batches
    args.save_every = 0 if args.save_every is None else args.save_every
    if args.run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{args.profile}-{stamp}"
    return args


def validate_args(args: argparse.Namespace) -> None:
    assert args.config.is_file(), f"Config not found: {args.config}"
    assert args.batch_size > 0, "--batch-size must be positive"
    assert args.seq_len > 1, "--seq-len must be at least 2"
    assert args.train_shards >= 0, "--train-shards must be non-negative"
    assert args.max_steps > 0, "--max-steps must be positive"
    assert args.log_every > 0, "--log-every must be positive"
    assert args.eval_every >= 0, "--eval-every must be non-negative"
    assert args.eval_batches > 0, "--eval-batches must be positive"
    assert args.save_every >= 0, "--save-every must be non-negative"
    assert args.lr > 0, "--lr must be positive"
    assert args.weight_decay >= 0, "--weight-decay must be non-negative"
    assert args.grad_clip >= 0, "--grad-clip must be non-negative"
    assert args.jepa_dropout_rate is None or 0.0 <= args.jepa_dropout_rate <= 1.0, "--jepa-dropout-rate must be in [0, 1]"
    assert args.auxiliary_layer_start is None or args.auxiliary_layer_start >= 0, "--auxiliary-layer-start must be non-negative"
    assert args.auxiliary_layer_stride is None or args.auxiliary_layer_stride > 0, "--auxiliary-layer-stride must be positive"


def config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.jepa_dropout_rate is not None:
        overrides["jepa_dropout_rate"] = args.jepa_dropout_rate
    if args.auxiliary_layer_start is not None:
        overrides["auxiliary_layer_start"] = args.auxiliary_layer_start
    if args.auxiliary_layer_stride is not None:
        overrides["auxiliary_layer_stride"] = args.auxiliary_layer_stride
    return overrides


def train_config_path(args: argparse.Namespace) -> Path:
    cached = getattr(args, "_train_config_path", None)
    if cached is not None:
        return cached

    overrides = config_overrides(args)
    if not overrides:
        args._train_config_path = args.config
        return args._train_config_path

    with args.config.open("r", encoding="utf-8") as handle:
        config_values = yaml.safe_load(handle)
    assert isinstance(config_values, dict), f"Expected mapping config in {args.config}"
    config_values.update(overrides)

    depth = int(config_values["depth"])
    num_jepa_blocks = depth - 1
    start = int(config_values.get("auxiliary_layer_start", 0))
    stride = int(config_values.get("auxiliary_layer_stride", 1))
    assert 0 <= start <= num_jepa_blocks, "--auxiliary-layer-start must select a valid JEPA block boundary"
    assert stride > 0, "--auxiliary-layer-stride must be positive"

    config_path = args.out_dir / args.run_name / "autoresearch_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config_values, sort_keys=False), encoding="utf-8")
    args._train_config_path = config_path
    return args._train_config_path


def dataset_dir_for_variant(variant: str) -> str:
    if variant == "byte260":
        return "fineweb10B_byte260"
    assert variant.startswith("sp") and variant[2:].isdigit(), f"Unsupported variant {variant!r}"
    return f"fineweb10B_{variant}"


def resolve_dataset_root_local(dataset_root: Path | None, parameter_golf_root: Path, variant: str) -> Path:
    if dataset_root is not None:
        return dataset_root.expanduser().resolve()
    return parameter_golf_root.expanduser().resolve() / "data" / "datasets" / dataset_dir_for_variant(variant)


def list_split_shards_local(dataset_root: Path, split: str) -> list[Path]:
    pattern = f"fineweb_{split}_*.bin"
    shards = sorted(dataset_root.glob(pattern))
    assert shards, f"No {split} shards found in {dataset_root} matching {pattern}"
    return shards


def tokenizer_path_for_variant(parameter_golf_root: Path, variant: str) -> tuple[Path, int]:
    root = parameter_golf_root.expanduser().resolve()
    manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
    dataset_name = dataset_dir_for_variant(variant)
    dataset_by_name = {entry["name"]: entry for entry in manifest["datasets"]}
    tokenizer_by_name = {entry["name"]: entry for entry in manifest["tokenizers"]}
    dataset_entry = dataset_by_name[dataset_name]
    tokenizer_entry = tokenizer_by_name[dataset_entry["tokenizer_name"]]
    return (root / "data" / tokenizer_entry["model_path"]).resolve(), int(dataset_entry["vocab_size"])


def ensure_results_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS, delimiter="\t").writeheader()


def git_short_hash() -> str:
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.SubprocessError:
        return "unknown"


def build_train_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--config", str(train_config_path(args)),
        "--parameter-golf-root", str(args.parameter_golf_root),
        "--variant", args.variant,
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--train-shards", str(args.train_shards),
        "--max-steps", str(args.max_steps),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--optimizer", args.optimizer,
        "--grad-clip", str(args.grad_clip),
        "--log-every", str(args.log_every),
        "--eval-every", str(args.eval_every),
        "--eval-batches", str(args.eval_batches),
        "--save-every", str(args.save_every),
        "--run-name", args.run_name,
        "--out-dir", str(args.out_dir),
        "--wandb-project", args.wandb_project,
        "--wandb-mode", args.wandb_mode,
        "--dtype", args.dtype,
        "--seed", str(args.seed),
    ]
    if args.dataset_root is not None:
        argv.extend(["--dataset-root", str(args.dataset_root)])
    if args.device is not None:
        argv.extend(["--device", args.device])
    if args.torchao_float8:
        argv.extend(["--torchao-float8", "--torchao-float8-recipe", args.torchao_float8_recipe])
    if args.compile:
        argv.append("--compile")
    if args.save_every == 0:
        argv.append("--return-model")
    return argv


def build_sentencepiece_luts(tokenizer, vocab_size: int, device) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    import torch

    sp_vocab_size = int(tokenizer.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int32)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)

    # Keep out-of-range IDs (padding to table_size) as boundary tokens so a
    # synthetic boundary byte is never added for them during BPB accounting.
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)

    for token_id in range(sp_vocab_size):
        if tokenizer.is_control(token_id) or tokenizer.is_unknown(token_id) or tokenizer.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if tokenizer.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = tokenizer.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))

    return (
        torch.tensor(base_bytes_np, dtype=torch.int32, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def evaluate_bpb(model, *, tokenizer, eval_loader, device, step: int) -> float:
    import torch
    import torch.nn.functional as F

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
    )
    was_training = model.training
    model.eval()
    total_nats = 0.0
    total_bytes = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids, labels = (tensor.to(device) for tensor in batch)
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                step=step,
                compute_aux_losses=False,
                lambda_jepa_scale=0.0,
                beta_sigreg_scale=0.0,
            )
            # The loader already shifts labels so labels[:, t] is the next token
            # for logits[:, t]. Re-shifting here would score t -> t+2 and inflate BPB.
            logits = outputs["logits"]
            prev_ids = input_ids.reshape(-1)
            targets = labels.reshape(-1)
            loss_flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets, reduction="none")
            valid_mask = targets != -100
            if not bool(valid_mask.any()):
                continue
            valid_targets = targets[valid_mask]
            valid_prev_ids = prev_ids[valid_mask]
            token_bytes = base_bytes_lut[valid_targets].to(dtype=torch.int32)
            has_leading_space = has_leading_space_lut[valid_targets]
            prev_is_boundary = is_boundary_token_lut[valid_prev_ids]

            if input_ids.dim() == 2:
                # Avoid cross-row leakage when eval batches flatten multiple
                # independent sequences into one vector.
                sequence_start_mask = torch.zeros_like(targets, dtype=torch.bool)
                sequence_start_mask = sequence_start_mask.reshape(input_ids.shape)
                sequence_start_mask[:, 0] = True
                sequence_start_mask = sequence_start_mask.reshape(-1)
                prev_is_boundary = prev_is_boundary | sequence_start_mask[valid_mask]

            token_bytes += (
                has_leading_space
                & ~prev_is_boundary
            ).to(dtype=token_bytes.dtype)

            # Match the challenge reference exactly: all token loss contributes to
            # the numerator, while the denominator is total target bytes.
            total_nats += loss_flat[valid_mask].sum().item()
            total_bytes += token_bytes.to(torch.float64).sum().item()
    if was_training:
        model.train()
    assert total_bytes > 0, "BPB evaluation saw zero target bytes"
    return total_nats / (math.log(2.0) * total_bytes)


def append_results_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS, delimiter="\t").writerow(row)


def build_results_row(args: argparse.Namespace, train_result: dict[str, Any], eval_metrics: dict[str, float], val_bpb: float, checkpoint_path: Path | None) -> dict[str, Any]:
    train_metrics = train_result.get("last_train_metrics") or {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_short_hash(),
        "run_name": args.run_name,
        "profile": args.profile,
        "status": args.status,
        "description": args.description,
        "val_bpb": f"{val_bpb:.6f}",
        "eval_loss": f"{eval_metrics.get('eval/loss', float('nan')):.6f}",
        "eval_loss_lm": f"{eval_metrics.get('eval/loss_lm', float('nan')):.6f}",
        "eval_loss_jepa": f"{eval_metrics.get('eval/loss_jepa', float('nan')):.6f}",
        "eval_loss_sigreg": f"{eval_metrics.get('eval/loss_sigreg', float('nan')):.6f}",
        "train_loss": f"{train_metrics.get('train/loss', float('nan')):.6f}",
        "train_loss_lm": f"{train_metrics.get('train/loss_lm', float('nan')):.6f}",
        "train_loss_jepa": f"{train_metrics.get('train/loss_jepa', float('nan')):.6f}",
        "train_loss_sigreg": f"{train_metrics.get('train/loss_sigreg', float('nan')):.6f}",
        "lambda_jepa": f"{train_metrics.get('train/lambda_jepa', float('nan')):.6f}",
        "beta_sigreg": f"{train_metrics.get('train/beta_sigreg', float('nan')):.6f}",
        "jepa_aux_dropped": f"{train_metrics.get('train/jepa_aux_dropped', float('nan')):.6f}",
        "jepa_dropout_fraction": f"{train_result.get('jepa_dropout_fraction', float('nan')):.6f}",
        "peak_memory_gb": f"{train_result.get('peak_memory_mb', 0.0) / 1024.0:.3f}",
        "tokens_per_sec": f"{train_metrics.get('train/tokens_per_sec', float('nan')):.6f}",
        "tokens_processed": str(train_result.get("tokens_processed", "")),
        "num_steps": str(train_result.get("step", "")),
        "wall_seconds": f"{train_result.get('wall_seconds', float('nan')):.6f}",
        "device": train_result.get("device", args.device or "auto"),
        "dtype": train_result.get("compute_dtype", args.dtype),
        "optimizer": args.optimizer,
        "lr": f"{args.lr:.8f}",
        "weight_decay": f"{args.weight_decay:.8f}",
        "batch_size": str(args.batch_size),
        "seq_len": str(args.seq_len),
        "train_shards": str(args.train_shards),
        "eval_batches": str(args.eval_batches),
        "config": str(train_config_path(args)),
        "checkpoint": "" if checkpoint_path is None else str(checkpoint_path),
        "run_dir": str(train_result["run_dir"]),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("---")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_results_tsv(args.results_path)
    if args.validate_only:
        return {"mode": "validate", "trainer_argv": build_train_argv(args)}

    import sentencepiece as spm
    import torch

    from data.dataset_helpers import build_eval_dataloader
    from scripts.inspect_checkpoint import load_checkpoint
    from scripts.train_intertwined_hjepa import detect_compute_dtype, evaluate as trainer_evaluate, main as train_intertwined_main

    train_result = train_intertwined_main(build_train_argv(args))
    checkpoint_path = Path(train_result["run_dir"]) / "latest.pt"
    device = torch.device(args.device or train_result.get("device", "cpu"))
    if checkpoint_path.is_file():
        model, checkpoint, _missing_fields = load_checkpoint(checkpoint_path, device)
        checkpoint_for_row: Path | None = checkpoint_path
    else:
        assert "model" in train_result, f"Expected checkpoint at {checkpoint_path}"
        model = train_result["model"]
        checkpoint = {"step": train_result.get("step", args.max_steps)}
        checkpoint_for_row = None

    compute_dtype = detect_compute_dtype(device, args.dtype)
    dataset_root = resolve_dataset_root_local(args.dataset_root, args.parameter_golf_root, args.variant)
    tokenizer_path, _vocab_size = tokenizer_path_for_variant(args.parameter_golf_root, args.variant)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    eval_loader = build_eval_dataloader(
        list_split_shards_local(dataset_root, "val"),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_batches=args.eval_batches,
        pin_memory=device.type == "cuda",
    )
    step = int(checkpoint.get("step", args.max_steps))
    eval_metrics = trainer_evaluate(model, eval_loader, device=device, non_blocking=device.type == "cuda", step=step, compute_dtype=compute_dtype)
    val_bpb = evaluate_bpb(model, tokenizer=tokenizer, eval_loader=eval_loader, device=device, step=step)
    summary = {
        "run_name": args.run_name,
        "profile": args.profile,
        "status": args.status,
        "val_bpb": val_bpb,
        "eval_loss": eval_metrics["eval/loss"],
        "eval_loss_lm": eval_metrics["eval/loss_lm"],
        "eval_loss_jepa": eval_metrics["eval/loss_jepa"],
        "eval_loss_sigreg": eval_metrics["eval/loss_sigreg"],
        "peak_memory_gb": train_result.get("peak_memory_mb", 0.0) / 1024.0,
        "tokens_processed": train_result["tokens_processed"],
        "num_steps": train_result["step"],
        "wall_seconds": train_result.get("wall_seconds", 0.0),
        "checkpoint": "" if checkpoint_for_row is None else str(checkpoint_for_row),
        "run_dir": str(train_result["run_dir"]),
        "results_path": str(args.results_path),
    }
    print_summary(summary)
    if args.write_results:
        append_results_row(args.results_path, build_results_row(args, train_result, eval_metrics, val_bpb, checkpoint_for_row))
    return {"mode": "run", "summary": summary, "eval_metrics": eval_metrics, "val_bpb": val_bpb}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = apply_profile_defaults(parse_args(argv))
    validate_args(args)
    return run(args)


if __name__ == "__main__":
    main()
