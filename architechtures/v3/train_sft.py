from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


PROMPT_FIELDS = ("problem", "prompt", "question")
SOLUTION_FIELDS = ("qwen3-solution", "qwen3_solution", "solution", "answer")
REASONING_FIELDS = ("qwen3-reasoning", "qwen3_reasoning", "reasoning_trace")
NVIDIA_DATASET = "nvidia/Nemotron-Post-Training-Dataset-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TRL SFT baseline.")
    parser.add_argument("--train-file")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-split", default="chat")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument(
        "--output-mode",
        default="solution_only",
        choices=["solution_only", "thought_then_solution"],
    )
    parser.add_argument("--reasoning-filter", default="any", choices=["any", "on", "off"])
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--seed", type=int, default=7)
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
    args = parser.parse_args()
    if bool(args.train_file) == bool(args.dataset_name):
        parser.error("Provide exactly one of --train-file or --dataset-name.")
    return args


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return getattr(torch, requested)


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
    return "on" if text in {"1", "true", "yes", "on"} else "off" if text in {"", "0", "false", "no", "off"} else text


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.train_file:
        with Path(args.train_file).open("r", encoding="utf-8") as handle:
            return [
                json.loads(line)
                for i, line in enumerate(handle)
                if args.max_train_samples is None or i < args.max_train_samples
            ]
    if args.dataset_name == NVIDIA_DATASET:
        return load_nvidia_rows(args.dataset_split, args.max_train_samples)

    from datasets import load_dataset

    split = args.dataset_split
    if args.max_train_samples is not None:
        split = f"{split}[:{args.max_train_samples}]"
    dataset = load_dataset(args.dataset_name, split=split, token=os.environ.get("HF_TOKEN"))
    return [dict(row) for row in dataset]


def load_nvidia_rows(split: str, max_rows: int | None) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    filenames = sorted(
        path
        for path in list_repo_files(NVIDIA_DATASET, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        if path.startswith(f"data/{split}-") and path.endswith(".parquet")
    )
    if not filenames:
        raise ValueError(f"No parquet files found for dataset split {split!r}.")

    for filename in filenames:
        local_path = hf_hub_download(
            repo_id=NVIDIA_DATASET,
            repo_type="dataset",
            filename=filename,
            token=os.environ.get("HF_TOKEN"),
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


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> dict[str, list[int]] | None:
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=False, return_dict=True)
    input_ids = list(full["input_ids"])
    prompt_ids = list(prompt["input_ids"])
    if len(input_ids) > max_length:
        return None
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokens are not a prefix of the full conversation tokens.")
    return {
        "input_ids": input_ids,
        "attention_mask": list(full["attention_mask"]),
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    }


def build_dataset(tokenizer: Any, args: argparse.Namespace) -> tuple[list[dict[str, list[int]]], int, int]:
    categories = set(args.categories) if args.categories else None
    dataset: list[dict[str, list[int]]] = []
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


def build_peft_config(args: argparse.Namespace):
    if not args.use_peft:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_rows, skipped, dropped = build_dataset(tokenizer, args)
    print(f"Loaded {len(tokenized_rows)} tokenized SFT examples (skipped {skipped}, dropped {dropped}).")

    trainer = SFTTrainer(
        model=args.model_name,
        args=SFTConfig(
            output_dir=args.output_dir,
            max_length=args.max_length,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            learning_rate=args.lr,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            packing=False,
            report_to=[],
            model_init_kwargs={
                "dtype": resolve_torch_dtype(args.dtype),
                "attn_implementation": args.attn_implementation,
                "trust_remote_code": True,
            },
        ),
        train_dataset=Dataset.from_list(tokenized_rows),
        processing_class=tokenizer,
        peft_config=build_peft_config(args),
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
