#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")


def dataset_dir_for_variant(variant: str) -> str:
    if variant == "byte260":
        return "fineweb10B_byte260"
    if variant.startswith("sp") and variant[2:].isdigit():
        return f"fineweb10B_{variant}"
    raise ValueError(f"Unsupported variant {variant!r}; expected byte260 or sp<VOCAB_SIZE>")


def read_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def read_shard_header(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=HEADER_DTYPE, count=HEADER_INTS)
    if header.size != HEADER_INTS:
        raise ValueError(f"{path} has a short header: expected {HEADER_INTS} ints, got {header.size}")
    return header


def describe_shard(path: Path, sample_tokens: int) -> dict:
    header = read_shard_header(path)
    num_tokens = int(header[2])
    header_bytes = HEADER_INTS * HEADER_DTYPE.itemsize
    expected_size = header_bytes + num_tokens * TOKEN_DTYPE.itemsize
    actual_size = path.stat().st_size
    sample = np.fromfile(
        path,
        dtype=TOKEN_DTYPE,
        count=sample_tokens,
        offset=header_bytes,
    )
    return {
        "path": str(path),
        "magic": int(header[0]),
        "version": int(header[1]),
        "num_tokens": num_tokens,
        "expected_size_bytes": expected_size,
        "actual_size_bytes": actual_size,
        "size_matches": expected_size == actual_size,
        "first_tokens": sample.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a cached Parameter-Golf FineWeb export")
    parser.add_argument(
        "--parameter-golf-root",
        default="/Users/namanchetwani/Projects/parameter-golf",
        help="Path to the parameter-golf repo root",
    )
    parser.add_argument(
        "--variant",
        default="sp1024",
        help="Dataset/tokenizer variant, e.g. sp1024 or byte260",
    )
    parser.add_argument(
        "--sample-tokens",
        type=int,
        default=12,
        help="How many payload tokens to print from each inspected shard",
    )
    parser.add_argument(
        "--max-train-shards",
        type=int,
        default=3,
        help="How many train shard headers to inspect",
    )
    parser.add_argument(
        "--max-val-shards",
        type=int,
        default=2,
        help="How many val shard headers to inspect",
    )
    args = parser.parse_args()

    root = Path(args.parameter_golf_root).expanduser().resolve()
    data_root = root / "data"
    manifest_path = data_root / "manifest.json"
    manifest = read_manifest(manifest_path)

    dataset_name = dataset_dir_for_variant(args.variant)
    dataset_root = data_root / "datasets" / dataset_name
    tokenizer_dir = data_root / "tokenizers"

    dataset_entry = next(
        (entry for entry in manifest.get("datasets", []) if entry.get("name") == dataset_name),
        None,
    )
    if dataset_entry is None:
        raise ValueError(f"Dataset {dataset_name!r} not found in {manifest_path}")

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next(
        (entry for entry in manifest.get("tokenizers", []) if entry.get("name") == tokenizer_name),
        None,
    )
    if tokenizer_entry is None:
        raise ValueError(f"Tokenizer {tokenizer_name!r} not found in {manifest_path}")

    train_shards = sorted(dataset_root.glob("fineweb_train_*.bin"))
    val_shards = sorted(dataset_root.glob("fineweb_val_*.bin"))

    print("Cached FineWeb Inspection")
    print(f"parameter_golf_root: {root}")
    print(f"manifest_path:        {manifest_path}")
    print(f"dataset_root:         {dataset_root}")
    print(f"tokenizer_dir:        {tokenizer_dir}")
    print()

    print("Manifest Summary")
    print(f"version:              {manifest.get('version')}")
    print(f"num_docs:             {manifest.get('num_docs')}")
    print(f"num_val_docs:         {manifest.get('num_val_docs')}")
    print(f"shuffle_seed:         {manifest.get('shuffle_seed')}")
    print(f"shard_size:           {manifest.get('shard_size')}")
    print(f"append_eos:           {manifest.get('append_eos')}")
    print()

    print("Dataset Entry")
    print(json.dumps(dataset_entry, indent=2))
    print()

    print("Tokenizer Entry")
    print(json.dumps(tokenizer_entry, indent=2))
    print()

    print("Resolved Tokenizer Assets")
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            resolved = data_root / value
            print(f"{key}: {resolved} {'[exists]' if resolved.exists() else '[missing]'}")
    print()

    print("Shard Inventory")
    print(f"train_shards_found:   {len(train_shards)}")
    print(f"val_shards_found:     {len(val_shards)}")
    print()

    print("Train Shard Samples")
    for shard in train_shards[: args.max_train_shards]:
        info = describe_shard(shard, args.sample_tokens)
        print(json.dumps(info, indent=2))
        if info["magic"] != SHARD_MAGIC or info["version"] != SHARD_VERSION:
            print("WARNING: unexpected shard header values")
    print()

    print("Val Shard Samples")
    for shard in val_shards[: args.max_val_shards]:
        info = describe_shard(shard, args.sample_tokens)
        print(json.dumps(info, indent=2))
        if info["magic"] != SHARD_MAGIC or info["version"] != SHARD_VERSION:
            print("WARNING: unexpected shard header values")


if __name__ == "__main__":
    main()
