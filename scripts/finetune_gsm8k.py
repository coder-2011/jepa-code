from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "llm-jepa" / "datasets" / "gsm8k_train.jsonl"
DEFAULT_TEST = ROOT / "llm-jepa" / "datasets" / "gsm8k_test.jsonl"
DEFAULT_WORKDIR = ROOT / "tmp" / "gsm8k_mlx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare GSM8K for mlx-lm LoRA fine-tuning and optionally launch training. "
            "This follows the official mlx-lm workflow: write {train,valid,test}.jsonl, "
            "write a small YAML config, then run `python -m mlx_lm.lora --config ... --mask-prompt`."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN))
    parser.add_argument("--test-file", default=str(DEFAULT_TEST))
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--valid-size", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(args: argparse.Namespace) -> tuple[Path, Path]:
    workdir = Path(args.workdir)
    data_dir = workdir / "data"
    if workdir.exists() and args.overwrite:
        shutil.rmtree(workdir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(Path(args.train_file), args.train_limit)
    test_rows = load_jsonl(Path(args.test_file), args.test_limit)

    valid_size = min(args.valid_size, max(1, len(train_rows) // 20), len(train_rows) - 1)
    valid_rows = train_rows[-valid_size:]
    train_rows = train_rows[:-valid_size]
    if not train_rows:
        raise ValueError("valid_size consumed the entire training set")

    # Official mlx-lm LoRA docs support `chat` JSONL directly, and GSM8K rows in this repo already
    # use the expected OpenAI-style `messages` list.
    write_jsonl(data_dir / "train.jsonl", train_rows)
    write_jsonl(data_dir / "valid.jsonl", valid_rows)
    write_jsonl(data_dir / "test.jsonl", test_rows)

    config = {
        "model": args.model,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "batch_size": args.batch_size,
        "iters": args.iters,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "num_layers": args.num_layers,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "save_every": 50,
        "adapter_path": str(workdir / "adapters"),
        "val_batches": -1,
        "test": False,
        "grad_checkpoint": False,
        "lora_parameters": {
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
        },
    }

    config_path = workdir / "mlx_lora_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return data_dir, config_path


def mlx_lm_command(config_path: Path) -> list[str]:
    return [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path), "--mask-prompt"]


def ensure_mlx_lm_available() -> None:
    probe = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "mlx-lm training entrypoint is unavailable. Install it with: pip install \"mlx-lm[train]\""
        )


def main() -> None:
    args = parse_args()
    data_dir, config_path = prepare_dataset(args)

    command = mlx_lm_command(config_path)
    print(f"Prepared mlx-lm dataset in {data_dir}")
    print(f"Wrote mlx-lm config to {config_path}")
    print("Command:")
    print(" ".join(command))

    if args.prepare_only:
        return

    ensure_mlx_lm_available()
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
