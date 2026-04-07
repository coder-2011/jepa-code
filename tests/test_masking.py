import random

import torch

from text_jepa.masking import (
    find_word_spans,
    map_words_to_tokens,
    mask_text,
    mask_text_from_yaml,
    sample_word_blocks,
)
from text_jepa.tokenization import tokenize_text

from conftest import FakeTokenizer, write_test_config


def test_find_word_spans_basic_sentence():
    spans = find_word_spans("The quick brown fox")

    # Word spans are half-open character ranges.
    assert spans == [(0, 3), (4, 9), (10, 15), (16, 19)]


def test_find_word_spans_excludes_surrounding_punctuation():
    spans = find_word_spans("Hello, world! don't-stop")

    # Punctuation should not create its own words or attach to surrounding words.
    assert spans == [(0, 5), (7, 12), (14, 24)]


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

    # Each raw word should map to at least one token span in this simple tokenizer.
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

    # Keep the top-level masking contract stable because later JEPA modules will depend on it.
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

    # Every target position should be swapped to the mask token id on the context side.
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

    # The masker should not perturb visible context tokens.
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

    # Positions are just the sparse form of the dense boolean target mask.
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

    # Special tokens are marked explicitly and should never leak into the target mask.
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

    # Padding should be excluded even when max_length is larger than the text.
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

    # Block masking is discrete, so we only expect a reasonable neighborhood around the target.
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

    # The sampled word-space blocks should respect the configured upper bound.
    assert all(end - start <= 2 for start, end in output["masked_span_ranges_word"])


def test_sample_word_blocks_prefers_a_closer_budget_fit():
    word_to_tokens = [
        {"token_start": 0, "token_end": 1},
        {"token_start": 1, "token_end": 2},
        {"token_start": 2, "token_end": 3},
        {"token_start": 3, "token_end": 4},
    ]

    blocks = sample_word_blocks(
        word_to_tokens,
        target_token_budget=3,
        max_block_words=2,
        rng=random.Random(0),
    )

    # The improved sampler should be able to hit an easy exact fit here.
    masked_tokens = sum(
        word_to_tokens[index]["token_end"] - word_to_tokens[index]["token_start"]
        for start, end in blocks
        for index in range(start, end)
    )
    assert masked_tokens == 3


def test_mask_text_from_yaml_uses_config_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_test_config(config_path, max_length=10, mask_ratio=0.4, max_block_words=2)
    tokenizer = FakeTokenizer()

    output = mask_text_from_yaml(
        tokenizer,
        "The quick brown fox jumps",
        config_path,
        rng=random.Random(0),
    )

    # The YAML path should propagate max_length through the convenience wrapper.
    assert output["input_ids_full"].shape[0] == 10
