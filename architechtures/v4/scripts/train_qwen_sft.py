from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


PROMPT_FIELDS = ("problem", "prompt", "question")
SOLUTION_FIELDS = ("qwen3-solution", "qwen3_solution", "solution", "answer")
REASONING_FIELDS = ("qwen3-reasoning", "qwen3_reasoning", "reasoning_trace")
NVIDIA_DATASET = "nvidia/Nemotron-Post-Training-Dataset-v2"
IGNORE_INDEX = -100


@dataclass
class TokenizedSFTExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-model SFT for Qwen-style chat models.")
    parser.add_argument("--train-file")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="chat")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-optimizer-state", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    parser.add_argument(
        "--lr-schedule",
        default="linear",
        choices=["constant", "linear", "cosine"],
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument(
        "--output-mode",
        default="solution_only",
        choices=["solution_only", "thought_then_solution"],
    )
    parser.add_argument("--reasoning-filter", default="any", choices=["any", "on", "off"])
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--wandb-mode", default="disabled")
    args = parser.parse_args(argv)
    if bool(args.train_file) == bool(args.dataset_name):
        parser.error("Provide exactly one of --train-file or --dataset-name.")
    if args.batch_size < 1 or args.grad_accum_steps < 1:
        parser.error("--batch-size and --grad-accum-steps must be >= 1.")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be >= 1 when provided.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    return getattr(torch, requested)


def should_autocast(device: torch.device, dtype: torch.dtype) -> bool:
    return device.type in {"cuda", "mps"} and dtype != torch.float32


def first_text(row: dict[str, Any], field_names: tuple[str, ...]) -> str:
    for name in field_names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def normalize_reasoning(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value or "").strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return "off"
    if text in {"1", "true", "yes", "on"}:
        return "on"
    return text


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.train_file:
        with Path(args.train_file).open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
            if args.max_train_samples is not None:
                rows = rows[: args.max_train_samples]
            return rows
    if (
        args.dataset_name == NVIDIA_DATASET
        and getattr(args, "dataset_config", None) is None
        and "," not in str(args.dataset_split)
    ):
        return load_nvidia_rows(args.dataset_split, args.max_train_samples, args.hf_token)

    from datasets import load_dataset

    split_names = [part.strip() for part in str(args.dataset_split).split(",") if part.strip()]
    if not split_names:
        raise ValueError("No dataset split names provided.")
    per_split_limit = None
    if args.max_train_samples is not None:
        per_split_limit = math.ceil(args.max_train_samples / len(split_names))

    rows: list[dict[str, Any]] = []
    for split_name in split_names:
        split = split_name
        if per_split_limit is not None:
            split = f"{split_name}[:{per_split_limit}]"
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=split,
            token=args.hf_token or os.environ.get("HF_TOKEN"),
        )
        rows.extend(dict(row) for row in dataset)
        if args.max_train_samples is not None and len(rows) >= args.max_train_samples:
            return rows[: args.max_train_samples]
    return rows


def load_nvidia_rows(split: str, max_rows: int | None, hf_token: str | None) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    filenames = sorted(
        path
        for path in list_repo_files(NVIDIA_DATASET, repo_type="dataset", token=hf_token or os.environ.get("HF_TOKEN"))
        if path.startswith(f"data/{split}-") and path.endswith(".parquet")
    )
    if not filenames:
        raise ValueError(f"No parquet files found for dataset split {split!r}.")
    for filename in filenames:
        local_path = hf_hub_download(
            repo_id=NVIDIA_DATASET,
            repo_type="dataset",
            filename=filename,
            token=hf_token or os.environ.get("HF_TOKEN"),
        )
        rows.extend(pq.read_table(local_path).to_pylist())
        if max_rows is not None and len(rows) >= max_rows:
            return rows[:max_rows]
    return rows


def row_messages(row: dict[str, Any], output_mode: str) -> list[dict[str, str]] | None:
    messages = row.get("messages")
    if isinstance(messages, list):
        cleaned = [
            {
                "role": str(message.get("role", "")).strip().lower(),
                "content": str(message.get("content", "")).strip(),
            }
            for message in messages
            if isinstance(message, dict)
            and str(message.get("role", "")).strip()
            and str(message.get("content", "")).strip()
        ]
        if len(cleaned) >= 2 and cleaned[-1]["role"] == "assistant":
            return cleaned

    prompt = first_text(row, PROMPT_FIELDS)
    solution = first_text(row, SOLUTION_FIELDS)
    reasoning = first_text(row, REASONING_FIELDS)
    if not prompt or not solution:
        return None
    if output_mode == "thought_then_solution" and reasoning and normalize_reasoning(row.get("reasoning", "on")) != "off":
        solution = f"<thought>\n{reasoning}\n</thought>\n{solution}"
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": solution},
    ]


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> TokenizedSFTExample | None:
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=False, return_dict=True)
    input_ids = list(full["input_ids"])
    prompt_ids = list(prompt["input_ids"])
    if len(input_ids) > max_length:
        return None
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokens are not a prefix of the full conversation tokens.")
    return TokenizedSFTExample(
        input_ids=input_ids,
        attention_mask=list(full["attention_mask"]),
        labels=[IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    )


def build_dataset(tokenizer: Any, args: argparse.Namespace) -> tuple[list[TokenizedSFTExample], int, int]:
    categories = set(args.categories) if args.categories else None
    dataset: list[TokenizedSFTExample] = []
    skipped = 0
    dropped = 0
    for row in load_rows(args):
        if categories is not None and str(row.get("category", "")).strip() not in categories:
            continue
        if args.reasoning_filter != "any" and normalize_reasoning(row.get("reasoning", "on")) != args.reasoning_filter:
            continue
        messages = row_messages(row, args.output_mode)
        if messages is None:
            skipped += 1
            continue
        tokenized = tokenize_messages(tokenizer, messages, args.max_length)
        if tokenized is None:
            dropped += 1
            continue
        dataset.append(tokenized)
    if not dataset:
        raise ValueError("No tokenized examples survived filtering.")
    return dataset, skipped, dropped


def collate_examples(batch: list[TokenizedSFTExample], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence(
        [torch.tensor(example.input_ids, dtype=torch.long) for example in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [torch.tensor(example.attention_mask, dtype=torch.long) for example in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [torch.tensor(example.labels, dtype=torch.long) for example in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    step: int,
    tokens_processed: int,
    metrics_tail: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_to_save.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    trainer_state: dict[str, Any] = {
        "step": step,
        "tokens_processed": tokens_processed,
        "args": vars(args),
        "metrics_tail": metrics_tail[-32:],
        "rng_state": {
            "torch": torch.get_rng_state().cpu(),
            "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        },
    }
    if args.save_optimizer_state:
        trainer_state["optimizer"] = optimizer.state_dict()
        trainer_state["scheduler"] = scheduler.state_dict()
    torch.save(trainer_state, checkpoint_dir / "trainer_state.pt")
    return checkpoint_dir


def count_tokens(labels: torch.Tensor) -> int:
    return int(labels.ne(IGNORE_INDEX).sum().item())


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    compute_dtype = resolve_torch_dtype(args.dtype, device)

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        token=hf_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset, skipped, dropped = build_dataset(tokenizer, args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_examples(batch, tokenizer.pad_token_id),
    )
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_optimizer_steps = args.max_steps or math.ceil(steps_per_epoch * args.epochs / args.grad_accum_steps)
    if total_optimizer_steps < 1:
        raise ValueError("Training resolved to zero optimizer steps.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=compute_dtype if device.type != "cpu" else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        token=hf_token,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    if args.compile:
        model = torch.compile(model, mode=args.torch_compile_mode)

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = get_scheduler(
        name=args.lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and compute_dtype == torch.float16))
    autocast_enabled = should_autocast(device, compute_dtype)

    model.train()
    step = 0
    optimizer_steps = 0
    tokens_processed = 0
    running_loss = 0.0
    metrics_tail: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    last_log_time = start_time

    optimizer.zero_grad(set_to_none=True)
    while optimizer_steps < total_optimizer_steps:
        for batch in dataloader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            tokens_processed += count_tokens(batch["labels"])
            with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=autocast_enabled):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum_steps
            running_loss += float(loss.detach().item())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step += 1

            if step % args.grad_accum_steps != 0:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            current_loss = running_loss
            running_loss = 0.0
            elapsed = time.perf_counter() - start_time
            metrics = {
                "step": optimizer_steps,
                "micro_step": step,
                "loss": current_loss,
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
                "tokens_processed": tokens_processed,
                "tokens_per_second": tokens_processed / max(elapsed, 1e-6),
                "seconds_since_start": elapsed,
            }
            metrics_tail.append(metrics)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")

            if args.logging_steps and optimizer_steps % args.logging_steps == 0:
                interval = time.perf_counter() - last_log_time
                last_log_time = time.perf_counter()
                print(
                    f"step={optimizer_steps} loss={current_loss:.4f} lr={metrics['lr']:.3e} "
                    f"grad_norm={grad_norm:.3f} tokens={tokens_processed} interval_s={interval:.2f}"
                )

            if args.save_steps and optimizer_steps % args.save_steps == 0:
                save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, optimizer_steps, tokens_processed, metrics_tail, args)

            if optimizer_steps >= total_optimizer_steps:
                break

    final_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    trainer_state: dict[str, Any] = {
        "step": optimizer_steps,
        "micro_step": step,
        "tokens_processed": tokens_processed,
        "args": vars(args),
        "metrics_tail": metrics_tail[-32:],
    }
    if args.save_optimizer_state:
        trainer_state["optimizer"] = optimizer.state_dict()
        trainer_state["scheduler"] = scheduler.state_dict()
    torch.save(trainer_state, output_dir / "trainer_state.pt")

    result = {
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": str(compute_dtype).replace("torch.", ""),
        "train_examples": len(dataset),
        "skipped_examples": skipped,
        "dropped_examples": dropped,
        "optimizer_steps": optimizer_steps,
        "micro_steps": step,
        "tokens_processed": tokens_processed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "last_metrics": metrics_tail[-1] if metrics_tail else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
