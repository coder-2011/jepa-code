from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import median
from typing import Any

from transformers import AutoTokenizer

from scripts.train_qwen_sft import row_messages, tokenize_messages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and split post-training chat data for Qwen SFT experiments.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output-mode", default="solution_only", choices=["solution_only", "thought_then_solution"])
    parser.add_argument("--reasoning-filter", default="any", choices=["any", "on", "off"])
    parser.add_argument("--category")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--val-count", type=int)
    parser.add_argument("--seed-tag", default="qwen-posttrain-v1")
    return parser.parse_args(argv)


def normalize_reasoning(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value or "").strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return "off"
    if text in {"1", "true", "yes", "on"}:
        return "on"
    return text


def normalized_signature(messages: list[dict[str, str]]) -> str:
    packed = "\n".join(f"{message['role']}\t{message['content']}" for message in messages)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def deterministic_key(signature: str, seed_tag: str) -> str:
    return hashlib.sha256(f"{seed_tag}:{signature}".encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


EMPTY_THINK_RE = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)


def sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for idx, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()
        if role == "assistant":
            content = EMPTY_THINK_RE.sub("", content).strip()
        if idx == 0 and role == "system" and not content:
            continue
        cleaned.append({"role": role, "content": content})
    return [message for message in cleaned if message["content"]]


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_rows = [json.loads(line) for line in Path(args.input_file).open("r", encoding="utf-8")]

    cleaned: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    skipped_missing = 0
    skipped_duplicate = 0
    skipped_filtered = 0
    dropped_length = 0
    token_lengths: list[int] = []
    assistant_token_lengths: list[int] = []

    for raw_row in raw_rows:
        if args.category is not None and str(raw_row.get("category", "")).strip() != args.category:
            skipped_filtered += 1
            continue
        if args.reasoning_filter != "any" and normalize_reasoning(raw_row.get("reasoning", "on")) != args.reasoning_filter:
            skipped_filtered += 1
            continue

        messages = row_messages(raw_row, args.output_mode)
        if messages is None:
            skipped_missing += 1
            continue

        messages = sanitize_messages(messages)
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            skipped_missing += 1
            continue

        signature = normalized_signature(messages)
        if signature in seen_signatures:
            skipped_duplicate += 1
            continue

        tokenized = tokenize_messages(tokenizer, messages, args.max_length)
        if tokenized is None:
            dropped_length += 1
            continue

        labels = tokenized.labels if hasattr(tokenized, "labels") else tokenized["labels"]
        input_ids = tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized["input_ids"]
        assistant_tokens = sum(1 for label in labels if label != -100)
        token_lengths.append(len(input_ids))
        assistant_token_lengths.append(assistant_tokens)

        kept_row = dict(raw_row)
        kept_row["messages"] = messages
        kept_row["_meta"] = {
            "signature": signature,
            "token_count": len(input_ids),
            "assistant_token_count": assistant_tokens,
            "deterministic_key": deterministic_key(signature, args.seed_tag),
        }
        cleaned.append(kept_row)
        seen_signatures.add(signature)

    cleaned.sort(key=lambda row: row["_meta"]["deterministic_key"])
    if args.max_examples is not None:
        cleaned = cleaned[: args.max_examples]

    train_count = args.train_count if args.train_count is not None else int(len(cleaned) * 0.9)
    val_count = args.val_count if args.val_count is not None else len(cleaned) - train_count
    train_rows = cleaned[:train_count]
    val_rows = cleaned[train_count : train_count + val_count]

    def strip_meta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{key: value for key, value in row.items() if key != "_meta"} for row in rows]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(train_path, strip_meta(train_rows))
    write_jsonl(val_path, strip_meta(val_rows))

    summary = {
        "input_file": args.input_file,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "output_mode": args.output_mode,
        "reasoning_filter": args.reasoning_filter,
        "category": args.category,
        "raw_rows": len(raw_rows),
        "clean_rows": len(cleaned),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "skipped_missing": skipped_missing,
        "skipped_duplicate": skipped_duplicate,
        "skipped_filtered": skipped_filtered,
        "dropped_length": dropped_length,
        "token_count_min": min(token_lengths) if token_lengths else 0,
        "token_count_median": median(token_lengths) if token_lengths else 0,
        "token_count_max": max(token_lengths) if token_lengths else 0,
        "assistant_token_min": min(assistant_token_lengths) if assistant_token_lengths else 0,
        "assistant_token_median": median(assistant_token_lengths) if assistant_token_lengths else 0,
        "assistant_token_max": max(assistant_token_lengths) if assistant_token_lengths else 0,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
