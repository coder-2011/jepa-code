import random

import torch
import yaml

from text_jepa.masking import (
    find_word_spans,
    map_words_to_tokens,
    mask_text,
    mask_text_from_yaml,
)
from text_jepa.tokenization import tokenize_text

from conftest import FakeTokenizer


def write_config(path, max_length=12, mask_ratio=0.4, max_block_words=2):
    path.write_text(
        yaml.safe_dump(
            {
                "tokenizer": {
                    "model_name": "Qwen/Qwen3-0.6B",
                    "max_length": max_length,
                    "mask_token": "[MASK]",
                },
                "masking": {
                    "mask_ratio": mask_ratio,
                    "max_block_words": max_block_words,
                },
            }
        ),
        encoding="utf-8",
    )


def test_find_word_spans_basic_sentence():
    spans = find_word_spans("The quick brown fox")

    assert spans == [(0, 3), (4, 9), (10, 15), (16, 19)]


def test_map_words_to_tokens_produces_token_ranges():
    tokenizer = FakeTokenizer()
    encoded = tokenize_text(tokenizer, "The quick brown fox", max_length=8)
    word_spans = find_word_spans("The quick brown fox")

    word_to_tokens = map_words_to_tokens(
        word_spans,
        encoded["offset_mapping"],
        encoded["attention_mask"].tolist(),
        encoded["special_tokens_mask"],
    )

    assert len(word_to_tokens) == 4
    assert all(item["token_start"] < item["token_end"] for item in word_to_tokens)


def test_mask_text_returns_required_contract():
    tokenizer = FakeTokenizer()

    output = mask_text(
        tokenizer,
        "The quick brown fox jumps over the lazy dog",
        max_length=16,
        mask_ratio=0.4,
        max_block_words=2,
        rng=random.Random(0),
    )

    assert set(output) == {
        "input_ids_full",
        "input_ids_ctx",
        "attention_mask",
        "target_mask",
        "target_positions",
        "target_token_ids",
        "masked_span_ranges_word",
        "masked_span_ranges_token",
    }


def test_masked_positions_are_replaced_in_context():
    tokenizer = FakeTokenizer()
    output = mask_text(
        tokenizer,
        "The quick brown fox jumps over the lazy dog",
        max_length=16,
        mask_ratio=0.4,
        max_block_words=2,
        rng=random.Random(0),
    )

    assert torch.all(output["input_ids_ctx"][output["target_mask"]] == tokenizer.mask_token_id)


def test_unmasked_positions_are_unchanged():
    tokenizer = FakeTokenizer()
    output = mask_text(
        tokenizer,
        "The quick brown fox jumps over the lazy dog",
        max_length=16,
        mask_ratio=0.4,
        max_block_words=2,
        rng=random.Random(0),
    )

    assert torch.equal(
        output["input_ids_ctx"][~output["target_mask"]],
        output["input_ids_full"][~output["target_mask"]],
    )


def test_target_positions_match_target_mask():
    tokenizer = FakeTokenizer()
    output = mask_text(
        tokenizer,
        "The quick brown fox jumps over the lazy dog",
        max_length=16,
        mask_ratio=0.4,
        max_block_words=2,
        rng=random.Random(0),
    )

    expected = torch.nonzero(output["target_mask"], as_tuple=False).squeeze(-1)
    assert torch.equal(output["target_positions"], expected)


def test_no_special_tokens_are_masked():
    tokenizer = FakeTokenizer()
    encoded = tokenize_text(tokenizer, "The quick brown fox", max_length=8)
    output = mask_text(
        tokenizer,
        "The quick brown fox",
        max_length=8,
        mask_ratio=0.5,
        max_block_words=2,
        rng=random.Random(0),
    )

    special_positions = torch.tensor(encoded["special_tokens_mask"], dtype=torch.bool)
    assert not torch.any(output["target_mask"] & special_positions)


def test_no_padding_tokens_are_masked():
    tokenizer = FakeTokenizer()
    output = mask_text(
        tokenizer,
        "The quick",
        max_length=8,
        mask_ratio=0.5,
        max_block_words=2,
        rng=random.Random(0),
    )

    padding_positions = output["attention_mask"] == 0
    assert not torch.any(output["target_mask"] & padding_positions)


def test_mask_ratio_is_approximate_not_exact():
    tokenizer = FakeTokenizer()
    output = mask_text(
        tokenizer,
        "The quick brown fox jumps over the lazy dog",
        max_length=16,
        mask_ratio=0.33,
        max_block_words=2,
        rng=random.Random(0),
    )

    masked_count = int(output["target_mask"].sum().item())
    assert 2 <= masked_count <= 5


def test_two_word_blocks_are_used_at_most():
    tokenizer = FakeTokenizer()
    output = mask_text(
        tokenizer,
        "The quick brown fox jumps over the lazy dog",
        max_length=16,
        mask_ratio=0.4,
        max_block_words=2,
        rng=random.Random(0),
    )

    assert all(end - start <= 2 for start, end in output["masked_span_ranges_word"])


def test_mask_text_from_yaml_uses_config_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path, max_length=10, mask_ratio=0.4, max_block_words=2)
    tokenizer = FakeTokenizer()

    output = mask_text_from_yaml(
        tokenizer,
        "The quick brown fox jumps",
        config_path,
        rng=random.Random(0),
    )

    assert output["input_ids_full"].shape[0] == 10
