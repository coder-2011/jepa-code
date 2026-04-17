from __future__ import annotations

import argparse
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.dataset_helpers import (
    build_eval_dataloader,
    build_train_dataloader,
    list_split_shards,
    resolve_dataset_root,
    select_train_shards,
)
from intertwined_hjepa import IntertwinedConfig, IntertwinedHJEPA


class _DisabledWandbRun:
    def log(self, *_args, **_kwargs) -> None:
        return None

    def finish(self) -> None:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Intertwined H-JEPA on cached FineWeb shards")
    parser.add_argument("--config", default="intertwined_hjepa.yaml")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--parameter-golf-root", type=Path, default=None)
    parser.add_argument("--variant", default="sp1024")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--train-shards", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw8bit", "adamw4bit", "adamwfp8"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/intertwined_hjepa"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb-project", default="intertwined-hjepa")
    parser.add_argument("--wandb-mode", default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument("--torchao-float8", action="store_true")
    parser.add_argument(
        "--torchao-float8-recipe",
        default="tensorwise",
        choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"],
    )
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=None)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    assert args.batch_size > 0, "--batch-size must be positive"
    assert args.max_steps > 0, "--max-steps must be positive"
    assert args.lr > 0, "--lr must be positive"
    assert args.weight_decay >= 0, "--weight-decay must be non-negative"
    assert args.grad_clip >= 0, "--grad-clip must be non-negative"
    assert args.log_every > 0, "--log-every must be positive"
    assert args.eval_every >= 0, "--eval-every must be non-negative"
    assert args.eval_batches > 0, "--eval-batches must be positive"
    assert args.save_every >= 0, "--save-every must be non-negative"
    assert args.seq_len is None or args.seq_len > 0, "--seq-len must be positive"
    assert args.train_shards is None or args.train_shards >= 0, "--train-shards must be non-negative"
    assert args.resume is None or args.resume.is_file(), f"Resume checkpoint not found: {args.resume}"
    assert not args.torchao_float8 or args.device in {None, "cuda"}, "--torchao-float8 currently requires CUDA"
    assert not args.torchao_float8 or args.dtype in {"auto", "bfloat16"}, "--torchao-float8 expects bf16 compute"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_synchronize(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps":
        return torch.mps.synchronize
    return lambda: None


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def detect_compute_dtype(device: torch.device, requested: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def build_autocast_context(device: torch.device, compute_dtype: torch.dtype):
    enabled = device.type == "cuda" and compute_dtype in {torch.float16, torch.bfloat16}
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=compute_dtype)


def optimizer_step(optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler | None) -> None:
    if scaler is None:
        optimizer.step()
        return
    scaler.step(optimizer)
    scaler.update()


def should_drop_jepa_loss(jepa_dropout_rate: float) -> bool:
    return jepa_dropout_rate > 0.0 and random.random() < jepa_dropout_rate


def maybe_apply_torchao_float8(model: IntertwinedHJEPA, args: argparse.Namespace, device: torch.device) -> None:
    if not args.torchao_float8:
        return
    assert device.type == "cuda", "TorchAO float8 training currently requires CUDA"
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    config = Float8LinearConfig.from_recipe_name(args.torchao_float8_recipe)
    convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=lambda mod, _fqn: (
            isinstance(mod, torch.nn.Linear)
            and mod.in_features % 16 == 0
            and mod.out_features % 16 == 0
            and min(mod.in_features, mod.out_features) >= 16
        ),
    )


def build_optimizer(model: IntertwinedHJEPA, args: argparse.Namespace, device: torch.device) -> torch.optim.Optimizer:
    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.student_parameters(), fused=device.type == "cuda", **kwargs)
    from torchao.optim import AdamW4bit, AdamW8bit, AdamWFp8

    optimizer_cls = {
        "adamw8bit": AdamW8bit,
        "adamw4bit": AdamW4bit,
        "adamwfp8": AdamWFp8,
    }[args.optimizer]
    return optimizer_cls(model.student_parameters(), **kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> IntertwinedConfig:
    return IntertwinedConfig.from_yaml(config_path)


def merge_config(config: IntertwinedConfig, args: argparse.Namespace) -> IntertwinedConfig:
    return replace(config, max_length=config.max_length if args.seq_len is None else args.seq_len)


def init_wandb(args: argparse.Namespace, config: IntertwinedConfig) -> Any:
    if args.wandb_mode == "disabled":
        return _DisabledWandbRun()
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=asdict(config),
    )


def _safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _safe_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(inner) for inner in value]
    return value


def save_checkpoint(
    path: Path,
    *,
    model: IntertwinedHJEPA,
    optimizer: torch.optim.Optimizer,
    config: IntertwinedConfig,
    step: int,
    tokens_processed: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "step": step,
        "tokens_processed": tokens_processed,
        "args": _safe_value(vars(args)),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        payload["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: IntertwinedHJEPA,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    return checkpoint


def move_batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    non_blocking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, labels = batch
    return (
        input_ids.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


@torch.no_grad()
def evaluate(
    model: IntertwinedHJEPA,
    eval_loader,
    *,
    device: torch.device,
    non_blocking: bool,
    step: int,
    compute_dtype: torch.dtype,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_lm = 0.0
    total_jepa = 0.0
    total_sigreg = 0.0
    jepa_layer_totals = [0.0] * (model.config.depth - 1)
    sigreg_layer_totals = [0.0] * (model.config.depth - 1)
    num_batches = 0

    for batch in eval_loader:
        input_ids, labels = move_batch_to_device(batch, device=device, non_blocking=non_blocking)
        with build_autocast_context(device, compute_dtype):
            outputs = model(input_ids=input_ids, labels=labels, step=step)
        total_loss += outputs["loss"].item()
        total_lm += outputs["loss_main"].item()
        total_jepa += outputs["loss_jepa"].item()
        total_sigreg += outputs["loss_sigreg"].item()
        for index, loss_value in enumerate(outputs["loss_jepa_layers"]):
            jepa_layer_totals[index] += loss_value.item()
        for index, loss_value in enumerate(outputs["loss_sigreg_layers"]):
            sigreg_layer_totals[index] += loss_value.item()
        num_batches += 1

    assert num_batches > 0, "Evaluation loader produced no batches"
    if was_training:
        model.train()

    denom = float(num_batches)
    metrics = {
        "eval/loss": total_loss / denom,
        "eval/loss_lm": total_lm / denom,
        "eval/loss_jepa": total_jepa / denom,
        "eval/loss_sigreg": total_sigreg / denom,
    }
    metrics.update(
        {
            f"eval/loss_jepa_layer_{index}": value / denom
            for index, value in enumerate(jepa_layer_totals)
        }
    )
    metrics.update(
        {
            f"eval/loss_sigreg_layer_{index}": value / denom
            for index, value in enumerate(sigreg_layer_totals)
        }
    )
    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    config = merge_config(load_config(Path(args.config)), args)
    device = torch.device(args.device or default_device())
    compute_dtype = detect_compute_dtype(device, args.dtype)
    compile_model = args.compile_model if args.compile_model is not None else device.type == "cuda"
    non_blocking = device.type == "cuda"
    synchronize = build_synchronize(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    dataset_root = resolve_dataset_root(args.dataset_root, args.parameter_golf_root, args.variant)
    train_shards = select_train_shards(list_split_shards(dataset_root, "train"), args.train_shards)
    val_shards = list_split_shards(dataset_root, "val")
    assert train_shards, "No train shards selected"

    model = IntertwinedHJEPA(config).to(device)
    maybe_apply_torchao_float8(model, args, device)
    optimizer = build_optimizer(model, args, device)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and compute_dtype == torch.float16 else None
    train_model = torch.compile(model, dynamic=False) if compile_model else model
    train_loader = build_train_dataloader(
        train_shards,
        batch_size=args.batch_size,
        seq_len=config.max_length,
        pin_memory=non_blocking,
    )
    eval_loader = None
    if args.eval_every > 0:
        eval_loader = build_eval_dataloader(
            val_shards,
            batch_size=args.batch_size,
            seq_len=config.max_length,
            max_batches=args.eval_batches,
            pin_memory=non_blocking,
        )

    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, config)

    start_step = 0
    tokens_processed = 0
    jepa_dropout_steps = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model=model, optimizer=optimizer, device=device)
        start_step = int(checkpoint["step"])
        tokens_processed = int(checkpoint["tokens_processed"])

    train_model.train()
    train_iter = iter(train_loader)
    input_ids, labels = move_batch_to_device(next(train_iter), device=device, non_blocking=non_blocking)
    last_train_metrics: dict[str, float] | None = None
    last_eval_metrics: dict[str, float] | None = None
    train_start_time = time.time()

    for step_index in range(start_step, args.max_steps):
        step_number = step_index + 1
        last_step = step_number == args.max_steps
        should_log = step_index % args.log_every == 0
        should_eval = args.eval_every > 0 and (step_number % args.eval_every == 0 or last_step)
        should_save = args.save_every > 0 and (step_number % args.save_every == 0 or last_step)

        synchronize()
        t0 = time.time()

        drop_jepa_loss = should_drop_jepa_loss(config.jepa_dropout_rate)
        jepa_dropout_steps += int(drop_jepa_loss)
        with build_autocast_context(device, compute_dtype):
            outputs = train_model(
                input_ids=input_ids,
                labels=labels,
                step=step_index,
                compute_aux_losses=not drop_jepa_loss,
            )
        loss = outputs["loss"]
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if last_step:
            next_input_ids = None
            next_labels = None
        else:
            next_input_ids, next_labels = move_batch_to_device(
                next(train_iter),
                device=device,
                non_blocking=non_blocking,
            )

        grad_norm = None
        if scaler is not None:
            scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.student_parameters(), args.grad_clip)

        optimizer_step(optimizer, scaler)
        model.update_ema(step_index)
        model.zero_grad(set_to_none=True)

        synchronize()
        step_time = time.time() - t0
        tokens_processed += input_ids.numel()

        if should_log:
            metrics = {
                "train/loss": loss.item(),
                "train/loss_lm": outputs["loss_main"].item(),
                "train/loss_jepa": outputs["loss_jepa"].item(),
                "train/loss_sigreg": outputs["loss_sigreg"].item(),
                "train/lambda_jepa": float(outputs["diagnostics"]["lambda_jepa"]),
                "train/beta_sigreg": float(outputs["diagnostics"]["beta_sigreg"]),
                "train/jepa_aux_dropped": float(drop_jepa_loss),
                "train/jepa_dropout_steps": float(jepa_dropout_steps),
                "train/jepa_dropout_fraction": jepa_dropout_steps / step_number,
                "train/step_time": step_time,
                "train/tokens_per_sec": input_ids.numel() / max(step_time, 1e-8),
            }
            if grad_norm is not None:
                metrics["train/grad_norm"] = float(grad_norm)
            metrics.update(
                {f"train/loss_jepa_layer_{index}": loss_value.item() for index, loss_value in enumerate(outputs["loss_jepa_layers"])}
            )
            metrics.update(
                {f"train/loss_sigreg_layer_{index}": loss_value.item() for index, loss_value in enumerate(outputs["loss_sigreg_layers"])}
            )
            metrics.update(
                {f"train/z_variance_layer_{index}": float(var_value) for index, var_value in enumerate(outputs["diagnostics"]["z_variance"])}
            )
            metrics.update(
                {f"train/z_std_mean_layer_{index}": float(std_value) for index, std_value in enumerate(outputs["diagnostics"]["z_std_mean"])}
            )
            metrics.update(
                {f"train/z_std_min_layer_{index}": float(std_value) for index, std_value in enumerate(outputs["diagnostics"]["z_std_min"])}
            )
            metrics.update(
                {f"train/delta_norm_layer_{index}": float(norm_value) for index, norm_value in enumerate(outputs["diagnostics"]["delta_norm"])}
            )
            print(
                f"step={step_index} tokens={tokens_processed} "
                + " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
            )
            wandb_run.log(metrics, step=step_index)
            last_train_metrics = metrics.copy()

        if should_eval:
            assert eval_loader is not None, "Evaluation requested without an eval loader"
            eval_metrics = evaluate(
                train_model,
                eval_loader,
                device=device,
                non_blocking=non_blocking,
                step=step_index,
                compute_dtype=compute_dtype,
            )
            print(
                f"eval step={step_index} "
                + " ".join(f"{key}={value:.4f}" for key, value in sorted(eval_metrics.items()))
            )
            wandb_run.log(eval_metrics, step=step_index)
            last_eval_metrics = eval_metrics.copy()

        if should_save:
            checkpoint_kwargs = {
                "model": model,
                "optimizer": optimizer,
                "config": config,
                "step": step_number,
                "tokens_processed": tokens_processed,
                "args": args,
            }
            save_checkpoint(run_dir / f"step-{step_number:06d}.pt", **checkpoint_kwargs)
            save_checkpoint(run_dir / "latest.pt", **checkpoint_kwargs)

        if not last_step:
            input_ids, labels = next_input_ids, next_labels

    wandb_run.finish()
    steps_completed = max(args.max_steps - start_step, 0)
    total_wall_seconds = time.time() - train_start_time
    return {
        "run_dir": run_dir,
        "step": args.max_steps,
        "tokens_processed": tokens_processed,
        "jepa_dropout_steps": jepa_dropout_steps,
        "jepa_dropout_fraction": jepa_dropout_steps / steps_completed if steps_completed > 0 else 0.0,
        "last_train_metrics": last_train_metrics,
        "last_eval_metrics": last_eval_metrics,
        "peak_memory_mb": peak_memory_mb(device),
        "wall_seconds": total_wall_seconds,
        "device": device.type,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
