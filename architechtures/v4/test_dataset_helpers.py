from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset_helpers import (
    build_eval_dataloader,
    build_train_dataloader,
    dataset_dir_for_variant,
    FineWebTokenLoader,
    HEADER_BYTES,
    HEADER_DTYPE,
    HEADER_INTS,
    resolve_dataset_root,
    SHARD_MAGIC,
    SHARD_VERSION,
    TOKEN_DTYPE,
    TokenStream,
    list_split_shards,
    load_data_shard,
    load_validation_tokens,
    select_train_shards,
)


def write_shard(path: Path, tokens: list[int], *, magic: int = SHARD_MAGIC, version: int = SHARD_VERSION):
    header = np.zeros((HEADER_INTS,), dtype=HEADER_DTYPE)
    header[0] = magic
    header[1] = version
    header[2] = len(tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        header.tofile(handle)
        np.asarray(tokens, dtype=TOKEN_DTYPE).tofile(handle)


def test_list_split_shards_and_select_train_shards(tmp_path: Path):
    dataset_root = tmp_path / "fineweb10B_sp1024"
    write_shard(dataset_root / "fineweb_train_000001.bin", [4, 5])
    write_shard(dataset_root / "fineweb_train_000000.bin", [1, 2, 3])
    write_shard(dataset_root / "fineweb_val_000000.bin", [9, 10])

    train_shards = list_split_shards(dataset_root, "train")
    val_shards = list_split_shards(dataset_root, "val")

    assert [path.name for path in train_shards] == [
        "fineweb_train_000000.bin",
        "fineweb_train_000001.bin",
    ]
    assert [path.name for path in val_shards] == ["fineweb_val_000000.bin"]

    assert select_train_shards(train_shards, None) == train_shards
    assert select_train_shards(train_shards, 0) == []
    assert select_train_shards(train_shards, 1) == train_shards[:1]


def test_load_data_shard_reads_tokens_and_validates_header(tmp_path: Path):
    shard_path = tmp_path / "fineweb_train_000000.bin"
    tokens = [1, 896, 319, 943, 956]
    write_shard(shard_path, tokens)

    loaded = load_data_shard(shard_path)

    assert loaded.dtype == torch.uint16
    assert loaded.tolist() == tokens


def test_load_data_shard_rejects_bad_magic(tmp_path: Path):
    shard_path = tmp_path / "fineweb_train_000000.bin"
    write_shard(shard_path, [1, 2, 3], magic=123)

    with pytest.raises(AssertionError, match="Unexpected shard header"):
        load_data_shard(shard_path)


def test_load_data_shard_rejects_size_mismatch(tmp_path: Path):
    shard_path = tmp_path / "fineweb_train_000000.bin"
    write_shard(shard_path, [1, 2, 3, 4])
    with shard_path.open("r+b") as handle:
        handle.truncate(HEADER_BYTES + 3 * TOKEN_DTYPE.itemsize)

    with pytest.raises(AssertionError, match="Shard size mismatch"):
        load_data_shard(shard_path)


def test_token_stream_take_crosses_shards_and_wraps(tmp_path: Path):
    shard_a = tmp_path / "fineweb_train_000000.bin"
    shard_b = tmp_path / "fineweb_train_000001.bin"
    shard_c = tmp_path / "fineweb_train_000002.bin"
    write_shard(shard_a, [10, 11, 12])
    write_shard(shard_b, [20, 21])
    write_shard(shard_c, [30, 31, 32, 33])

    stream = TokenStream([shard_a, shard_b, shard_c])

    first = stream.take(2)
    second = stream.take(4)
    third = stream.take(5)
    fourth = stream.take(3)

    assert first.tolist() == [10, 11]
    assert second.tolist() == [12, 20, 21, 30]
    assert third.tolist() == [31, 32, 33, 10, 11]
    assert fourth.tolist() == [12, 20, 21]


def test_fineweb_token_loader_builds_shifted_batches_across_boundaries(tmp_path: Path):
    shard_a = tmp_path / "fineweb_train_000000.bin"
    shard_b = tmp_path / "fineweb_train_000001.bin"
    write_shard(shard_a, [1, 2, 3, 4, 5])
    write_shard(shard_b, [6, 7, 8, 9, 10])

    loader = FineWebTokenLoader([shard_a, shard_b], device="cpu")

    x1, y1 = loader.next_batch(batch_size=2, seq_len=2)
    x2, y2 = loader.next_batch(batch_size=2, seq_len=2)
    x3, y3 = loader.next_batch(batch_size=2, seq_len=2)

    assert x1.tolist() == [[1, 2], [3, 4]]
    assert y1.tolist() == [[2, 3], [4, 5]]

    # Batches are disjoint chunks from the token stream; the one-token shift happens
    # inside each batch, not across consecutive next_batch calls.
    assert x2.tolist() == [[6, 7], [8, 9]]
    assert y2.tolist() == [[7, 8], [9, 10]]

    # Third batch wraps to the first shard and still preserves the internal 1-token shift.
    assert x3.tolist() == [[1, 2], [3, 4]]
    assert y3.tolist() == [[2, 3], [4, 5]]

    assert x1.dtype == torch.int64
    assert y1.dtype == torch.int64


def test_load_validation_tokens_concatenates_and_trims_to_seq_len(tmp_path: Path):
    shard_a = tmp_path / "fineweb_val_000000.bin"
    shard_b = tmp_path / "fineweb_val_000001.bin"
    write_shard(shard_a, [1, 2, 3, 4, 5, 6])
    write_shard(shard_b, [7, 8, 9, 10])

    tokens = load_validation_tokens([shard_a, shard_b], seq_len=4)

    # Total tokens = 10, so usable = ((10 - 1) // 4) * 4 = 8, return usable + 1 = 9
    assert tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert (tokens.numel() - 1) % 4 == 0


def test_build_train_dataloader_streams_prebatched_examples(tmp_path: Path):
    shard_a = tmp_path / "fineweb_train_000000.bin"
    shard_b = tmp_path / "fineweb_train_000001.bin"
    write_shard(shard_a, [1, 2, 3, 4, 5])
    write_shard(shard_b, [6, 7, 8, 9, 10])

    loader = build_train_dataloader([shard_a, shard_b], batch_size=2, seq_len=2)
    iterator = iter(loader)
    x1, y1 = next(iterator)
    x2, y2 = next(iterator)

    assert x1.tolist() == [[1, 2], [3, 4]]
    assert y1.tolist() == [[2, 3], [4, 5]]
    assert x2.tolist() == [[6, 7], [8, 9]]
    assert y2.tolist() == [[7, 8], [9, 10]]


def test_build_eval_dataloader_uses_tensor_views_and_batching(tmp_path: Path):
    shard_a = tmp_path / "fineweb_val_000000.bin"
    shard_b = tmp_path / "fineweb_val_000001.bin"
    write_shard(shard_a, [1, 2, 3, 4, 5, 6])
    write_shard(shard_b, [7, 8, 9, 10, 11, 12])

    loader = build_eval_dataloader([shard_a, shard_b], batch_size=2, seq_len=2, max_batches=2)
    batches = list(loader)

    assert len(batches) == 2
    assert batches[0][0].tolist() == [[1, 2], [3, 4]]
    assert batches[0][1].tolist() == [[2, 3], [4, 5]]
    assert batches[1][0].tolist() == [[5, 6], [7, 8]]
    assert batches[1][1].tolist() == [[6, 7], [8, 9]]


def test_dataset_root_resolution_helpers(tmp_path: Path):
    dataset_root = tmp_path / "custom"
    parameter_golf_root = tmp_path / "parameter-golf"

    assert dataset_dir_for_variant("byte260") == "fineweb10B_byte260"
    assert dataset_dir_for_variant("sp1024") == "fineweb10B_sp1024"
    assert resolve_dataset_root(dataset_root, None, "sp1024") == dataset_root.resolve()
    assert resolve_dataset_root(None, parameter_golf_root, "sp1024") == (
        parameter_golf_root.resolve() / "data" / "datasets" / "fineweb10B_sp1024"
    )


def test_loader_asserts_on_invalid_arguments(tmp_path: Path):
    shard_path = tmp_path / "fineweb_train_000000.bin"
    write_shard(shard_path, [1, 2, 3, 4])

    with pytest.raises(AssertionError, match="Unsupported split"):
        list_split_shards(tmp_path, "test")
    with pytest.raises(AssertionError, match="non-negative"):
        select_train_shards([shard_path], -1)
    with pytest.raises(AssertionError, match="available"):
        select_train_shards([shard_path], 2)
    with pytest.raises(AssertionError, match="requires at least one shard"):
        TokenStream([])
    with pytest.raises(AssertionError, match="seq_len must be positive"):
        load_validation_tokens([shard_path], 0)
    with pytest.raises(AssertionError, match="batch_size must be positive"):
        FineWebTokenLoader([shard_path]).next_batch(0, 4)
    with pytest.raises(AssertionError, match="seq_len must be positive"):
        FineWebTokenLoader([shard_path]).next_batch(1, 0)
    with pytest.raises(AssertionError, match="Unsupported variant"):
        dataset_dir_for_variant("bad")
    with pytest.raises(AssertionError, match="Provide either --dataset-root or --parameter-golf-root"):
        resolve_dataset_root(None, None, "sp1024")
    with pytest.raises(AssertionError, match="max_batches must be positive"):
        build_eval_dataloader([shard_path], batch_size=1, seq_len=4, max_batches=0)
