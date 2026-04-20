from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")
HEADER_BYTES = HEADER_INTS * HEADER_DTYPE.itemsize


def _read_shard_header(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=HEADER_DTYPE, count=HEADER_INTS)
    assert header.size == HEADER_INTS, f"Short header in shard {path}"
    assert int(header[0]) == SHARD_MAGIC and int(header[1]) == SHARD_VERSION, (
        f"Unexpected shard header in {path}: magic={int(header[0])} version={int(header[1])}"
    )
    return header


def _expected_shard_size(num_tokens: int) -> int:
    return HEADER_BYTES + num_tokens * TOKEN_DTYPE.itemsize


def dataset_dir_for_variant(variant: str) -> str:
    if variant == "byte260":
        return "fineweb10B_byte260"
    assert variant.startswith("sp") and variant[2:].isdigit(), (
        f"Unsupported variant {variant!r}; expected byte260 or sp<VOCAB_SIZE>"
    )
    return f"fineweb10B_{variant}"


def resolve_dataset_root(
    dataset_root: Path | None,
    parameter_golf_root: Path | None,
    variant: str,
) -> Path:
    if dataset_root is not None:
        return dataset_root.expanduser().resolve()
    assert parameter_golf_root is not None, "Provide either --dataset-root or --parameter-golf-root"
    return (
        parameter_golf_root.expanduser().resolve()
        / "data"
        / "datasets"
        / dataset_dir_for_variant(variant)
    )


def list_split_shards(dataset_root: Path, split: str) -> list[Path]:
    assert split in {"train", "val"}, f"Unsupported split {split!r}; expected 'train' or 'val'"
    pattern = f"fineweb_{split}_*.bin"
    shards = sorted(dataset_root.glob(pattern))
    assert shards, f"No {split} shards found in {dataset_root} matching {pattern}"
    return shards


def select_train_shards(shards: list[Path], train_shards: int | None) -> list[Path]:
    if train_shards is None:
        return shards
    assert train_shards >= 0, "train_shards must be non-negative"
    assert train_shards <= len(shards), (
        f"Requested {train_shards} train shards, but only {len(shards)} are available"
    )
    return shards[:train_shards]


def load_data_shard(path: Path) -> torch.Tensor:
    header = _read_shard_header(path)
    num_tokens = int(header[2])
    expected_size = _expected_shard_size(num_tokens)
    actual_size = path.stat().st_size
    assert actual_size == expected_size, (
        f"Shard size mismatch for {path}: expected {expected_size}, got {actual_size}"
    )

    tokens_np = np.fromfile(path, dtype=TOKEN_DTYPE, count=num_tokens, offset=HEADER_BYTES)
    assert tokens_np.size == num_tokens, f"Short token read for {path}"

    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(shard_paths: list[Path], seq_len: int) -> torch.Tensor:
    assert seq_len > 0, "seq_len must be positive"
    tokens = torch.cat([load_data_shard(path) for path in shard_paths]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    assert usable > 0, f"Validation split is too short for seq_len={seq_len}"
    return tokens[: usable + 1]


class TokenStream:
    """Sequential token stream over one or more cached FineWeb shards."""

    def __init__(self, shard_paths: list[Path]):
        assert shard_paths, "TokenStream requires at least one shard"
        self.shard_paths = shard_paths
        self.shard_idx = 0
        self.tokens = load_data_shard(self.shard_paths[self.shard_idx])
        self.pos = 0

    def _advance_shard(self) -> None:
        self.shard_idx = (self.shard_idx + 1) % len(self.shard_paths)
        self.tokens = load_data_shard(self.shard_paths[self.shard_idx])
        self.pos = 0

    def take(self, n: int) -> torch.Tensor:
        assert n > 0, "take(n) requires n > 0"
        chunks: list[torch.Tensor] = []
        remaining = n
        while remaining > 0:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self._advance_shard()
                continue
            k = min(remaining, available)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class FineWebTokenLoader:
    """
    Simple sequential loader for cached FineWeb token shards.

    It reads from a continuous token stream and builds causal-LM batches:
    x = tokens[:-1], y = tokens[1:].
    """

    def __init__(self, shard_paths: list[Path], device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.stream = TokenStream(shard_paths)

    def next_batch(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch_size > 0, "batch_size must be positive"
        assert seq_len > 0, "seq_len must be positive"

        total_tokens = batch_size * seq_len
        local = self.stream.take(total_tokens + 1).to(dtype=torch.int64)
        input_ids = local[:-1].reshape(batch_size, seq_len)
        labels = local[1:].reshape(batch_size, seq_len)
        return (
            input_ids.to(self.device, non_blocking=True),
            labels.to(self.device, non_blocking=True),
        )


class FineWebTrainBatchDataset(IterableDataset):
    """Infinite stream of pre-batched causal-LM training examples."""

    def __init__(self, shard_paths: list[Path], batch_size: int, seq_len: int):
        assert shard_paths, "FineWebTrainBatchDataset requires at least one shard"
        assert batch_size > 0, "batch_size must be positive"
        assert seq_len > 0, "seq_len must be positive"
        self.shard_paths = shard_paths
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        stream = TokenStream(self.shard_paths)
        total_tokens = self.batch_size * self.seq_len
        while True:
            local = stream.take(total_tokens + 1).to(dtype=torch.int64)
            yield (
                local[:-1].reshape(self.batch_size, self.seq_len),
                local[1:].reshape(self.batch_size, self.seq_len),
            )


def build_train_dataloader(
    shard_paths: list[Path],
    *,
    batch_size: int,
    seq_len: int,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = FineWebTrainBatchDataset(shard_paths, batch_size=batch_size, seq_len=seq_len)
    return DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=pin_memory)


def build_eval_dataloader(
    shard_paths: list[Path],
    *,
    batch_size: int,
    seq_len: int,
    max_batches: int,
    pin_memory: bool = False,
) -> DataLoader:
    assert batch_size > 0, "batch_size must be positive"
    assert seq_len > 0, "seq_len must be positive"
    assert max_batches > 0, "max_batches must be positive"

    tokens = load_validation_tokens(shard_paths, seq_len=seq_len).to(dtype=torch.int64)
    inputs = tokens[:-1].view(-1, seq_len)
    labels = tokens[1:].view(-1, seq_len)
    num_examples = min(inputs.shape[0], batch_size * max_batches)
    num_examples -= num_examples % batch_size
    assert num_examples > 0, (
        f"Validation split is too short for batch_size={batch_size}, seq_len={seq_len}, max_batches={max_batches}"
    )
    dataset = TensorDataset(inputs[:num_examples], labels[:num_examples])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=pin_memory)
