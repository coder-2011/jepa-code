from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROMPT_FIELDS = ("problem", "prompt", "question")
SOLUTION_FIELDS = ("qwen3-solution", "qwen3_solution", "solution", "answer")
REASONING_FIELDS = ("qwen3-reasoning", "qwen3_reasoning", "reasoning_trace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TRL SFT baseline.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--output-mode",
        default="solution_only",
        choices=["solution_only", "thought_then_solution"],
    )
    parser.add_argument(
        "--reasoning-filter",
        default="any",
        choices=["any", "on", "off"],
    )
    parser.add_argument("--category", action="append", dest="categories", default=None)
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_reasoning_flag(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = normalize_text(value).lower()
    if text in {"1", "true", "yes", "on"}:
        return "on"
    if text in {"", "0", "false", "no", "off"}:
        return "off"
    return text


def first_text_field(row: dict[str, Any], field_names: tuple[str, ...]) -> str:
    for field_name in field_names:
        text = normalize_text(row.get(field_name))
        if text:
            return text
    return ""


def split_chat_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return [], ""

    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role")).lower()
        content = normalize_text(message.get("content"))
        if role and content:
            normalized_messages.append({"role": role, "content": content})

    for index in range(len(normalized_messages) - 1, -1, -1):
        if normalized_messages[index]["role"] == "assistant":
            return normalized_messages[:index], normalized_messages[index]["content"]
    return [], ""


def build_assistant_target(row: dict[str, Any], output_mode: str) -> str:
    solution = first_text_field(row, SOLUTION_FIELDS)
    reasoning = first_text_field(row, REASONING_FIELDS)
    reasoning_flag = normalize_reasoning_flag(row.get("reasoning", "on"))

    if output_mode == "solution_only" or not reasoning or reasoning_flag == "off":
        return solution
    return f"<thought>\n{reasoning}\n</thought>\n{solution}"


def should_keep_row(
    row: dict[str, Any],
    reasoning_filter: str,
    categories: set[str] | None,
) -> bool:
    if categories is not None and normalize_text(row.get("category")) not in categories:
        return False
    if reasoning_filter == "any":
        return True
    return normalize_reasoning_flag(row.get("reasoning", "on")) == reasoning_filter


def read_jsonl(path: str | Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def row_to_messages(
    row: dict[str, Any],
    output_mode: str,
) -> dict[str, list[dict[str, str]]] | None:
    prompt_messages, assistant_text = split_chat_messages(row)
    if prompt_messages and assistant_text:
        return {"messages": prompt_messages + [{"role": "assistant", "content": assistant_text}]}

    prompt = first_text_field(row, PROMPT_FIELDS)
    completion = build_assistant_target(row, output_mode)
    if not prompt or not completion:
        return None
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    }


def build_sft_records(args: argparse.Namespace) -> list[dict[str, list[dict[str, str]]]]:
    categories = set(args.categories) if args.categories else None
    records: list[dict[str, list[dict[str, str]]]] = []
    skipped = 0
    for row in read_jsonl(args.train_file, args.max_train_samples):
        if not should_keep_row(row, args.reasoning_filter, categories):
            continue
        record = row_to_messages(row, args.output_mode)
        if record is None:
            skipped += 1
            continue
        records.append(record)
    if not records:
        raise ValueError("No training examples survived filtering.")
    print(f"Loaded {len(records)} SFT examples (skipped {skipped}).")
    return records


def resolve_torch_dtype(requested: str):
    import torch

    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def tokenize_sft_record(
    tokenizer: Any,
    record: dict[str, list[dict[str, str]]],
    max_length: int,
) -> dict[str, list[int]] | None:
    messages = record["messages"]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Expected messages ending in a single assistant turn.")

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    prompt_encoded = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    prompt_input_ids = list(prompt_encoded["input_ids"])
    if len(input_ids) > max_length:
        return None
    if input_ids[: len(prompt_input_ids)] != prompt_input_ids:
        raise ValueError("Prompt tokens are not a prefix of the full conversation tokens.")
    labels = [-100] * len(prompt_input_ids) + input_ids[len(prompt_input_ids) :]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_tokenized_dataset(
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, list[int]]], int]:
    tokenized_rows: list[dict[str, list[int]]] = []
    skipped = 0
    for record in build_sft_records(args):
        tokenized = tokenize_sft_record(tokenizer, record, args.max_length)
        if tokenized is None:
            skipped += 1
            continue
        tokenized_rows.append(tokenized)
    if not tokenized_rows:
        raise ValueError("No tokenized examples survived max_length filtering.")
    return tokenized_rows, skipped


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Heavy HF imports stay after argument parsing so `--help` remains instant.
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoTokenizer

    torch_dtype = resolve_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_rows, dropped_for_length = build_tokenized_dataset(tokenizer, args)
    print(f"Tokenized {len(tokenized_rows)} rows (dropped {dropped_for_length} for max_length).")
    train_dataset = Dataset.from_list(tokenized_rows)

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
                "dtype": torch_dtype,
                "attn_implementation": args.attn_implementation,
                "trust_remote_code": True,
            },
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
