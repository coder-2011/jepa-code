import json

from text_jepa.data import FineWebJsonlDataset, create_fineweb_dataloader

from conftest import FakeTokenizer, write_test_config


def write_sample_jsonl(path):
    rows = [
        {
            "text": "The quick brown fox jumps over the lazy dog again and again.",
            "token_count": 128,
            "language_score": 0.99,
        },
        {
            "text": "Short boilerplate page.",
            "token_count": 32,
            "language_score": 0.99,
        },
        {
            "text": "This document is long enough but has a weak language score.",
            "token_count": 140,
            "language_score": 0.75,
        },
        {
            "text": "Pack my box with five dozen liquor jugs and some extra words for masking.",
            "token_count": 160,
            "language_score": 0.985,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_fineweb_jsonl_dataset_filters_docs(tmp_path):
    config_path = tmp_path / "config.yaml"
    data_path = tmp_path / "fineweb.jsonl"
    write_test_config(config_path, max_length=16, mask_ratio=0.4, max_block_words=2)
    write_sample_jsonl(data_path)

    dataset = FineWebJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeTokenizer(),
        config_path=config_path,
        min_token_count=128,
        min_language_score=0.98,
    )

    assert len(dataset) == 2


def test_fineweb_jsonl_dataset_masks_deterministically_per_index(tmp_path):
    config_path = tmp_path / "config.yaml"
    data_path = tmp_path / "fineweb.jsonl"
    write_test_config(config_path, max_length=16, mask_ratio=0.4, max_block_words=2)
    write_sample_jsonl(data_path)

    dataset_a = FineWebJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeTokenizer(),
        config_path=config_path,
        seed=7,
    )
    dataset_b = FineWebJsonlDataset(
        jsonl_path=data_path,
        tokenizer=FakeTokenizer(),
        config_path=config_path,
        seed=7,
    )

    example_a = dataset_a[0]
    example_b = dataset_b[0]

    assert example_a["input_ids_full"].equal(example_b["input_ids_full"])
    assert example_a["input_ids_ctx"].equal(example_b["input_ids_ctx"])
    assert example_a["target_positions"].equal(example_b["target_positions"])


def test_create_fineweb_dataloader_returns_masked_batch_contract(tmp_path):
    config_path = tmp_path / "config.yaml"
    data_path = tmp_path / "fineweb.jsonl"
    write_test_config(config_path, max_length=16, mask_ratio=0.4, max_block_words=2)
    write_sample_jsonl(data_path)

    dataloader = create_fineweb_dataloader(
        jsonl_path=data_path,
        tokenizer=FakeTokenizer(),
        config_path=config_path,
        batch_size=2,
        min_token_count=128,
        min_language_score=0.98,
    )

    batch = next(iter(dataloader))

    assert batch["input_ids_full"].shape == (2, 16)
    assert batch["input_ids_ctx"].shape == (2, 16)
    assert batch["attention_mask"].shape == (2, 16)
    assert batch["target_mask"].shape == (2, 16)
    assert batch["target_positions"].shape[0] == 2
    assert batch["target_valid_mask"].shape == batch["target_positions"].shape
