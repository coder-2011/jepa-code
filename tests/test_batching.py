import random

import torch

from text_jepa.batching import collate_masked_examples
from text_jepa.masking import mask_text

from conftest import FakeTokenizer


def make_masked_example(text, max_length=16, mask_ratio=0.4, max_block_words=2, seed=0):
    tokenizer = FakeTokenizer()
    return mask_text(
        tokenizer,
        text,
        max_length=max_length,
        mask_ratio=mask_ratio,
        max_block_words=max_block_words,
        rng=random.Random(seed),
    )


def test_collate_stacks_fixed_length_sequence_tensors():
    examples = [
        make_masked_example("The quick brown fox", seed=0),
        make_masked_example("Jumps over the lazy dog", seed=1),
    ]

    batch = collate_masked_examples(examples)

    assert batch["input_ids_full"].shape == (2, 16)
    assert batch["input_ids_ctx"].shape == (2, 16)
    assert batch["attention_mask"].shape == (2, 16)
    assert batch["target_mask"].shape == (2, 16)


def test_collate_pads_target_positions_to_t_max():
    examples = [
        make_masked_example("The quick brown fox", seed=0),
        make_masked_example("The quick brown fox jumps over the lazy dog", seed=1),
    ]

    batch = collate_masked_examples(examples)
    t_max = max(example["target_positions"].shape[0] for example in examples)

    assert batch["target_positions"].shape == (2, t_max)
    assert batch["target_token_ids"].shape == (2, t_max)


def test_collate_builds_target_valid_mask():
    examples = [
        make_masked_example("The quick brown fox", seed=0),
        make_masked_example("The quick brown fox jumps over the lazy dog", seed=1),
    ]

    batch = collate_masked_examples(examples)

    for batch_index, example in enumerate(examples):
        target_count = example["target_positions"].shape[0]
        assert torch.all(batch["target_valid_mask"][batch_index, :target_count])
        assert not torch.any(batch["target_valid_mask"][batch_index, target_count:])


def test_collate_preserves_target_token_ids():
    examples = [
        make_masked_example("The quick brown fox", seed=0),
        make_masked_example("The quick brown fox jumps over the lazy dog", seed=1),
    ]

    batch = collate_masked_examples(examples)

    for batch_index, example in enumerate(examples):
        target_count = example["target_token_ids"].shape[0]
        assert torch.equal(
            batch["target_token_ids"][batch_index, :target_count],
            example["target_token_ids"],
        )


def test_collate_rejects_empty_batch():
    try:
        collate_masked_examples([])
    except ValueError as exc:
        assert "examples must be non-empty" in str(exc)
    else:
        raise AssertionError("Expected collate_masked_examples to reject an empty batch")


def test_collate_rejects_missing_required_key():
    example = make_masked_example("The quick brown fox", seed=0)
    del example["target_positions"]

    try:
        collate_masked_examples([example])
    except ValueError as exc:
        assert "example is missing keys" in str(exc)
    else:
        raise AssertionError("Expected collate_masked_examples to reject missing keys")


def test_collate_rejects_sequence_length_mismatch():
    examples = [
        make_masked_example("The quick brown fox", max_length=12, seed=0),
        make_masked_example("Jumps over the lazy dog", max_length=16, seed=1),
    ]

    try:
        collate_masked_examples(examples)
    except ValueError as exc:
        assert "same sequence length" in str(exc)
    else:
        raise AssertionError("Expected collate_masked_examples to reject length mismatch")


def test_collate_supports_zero_target_example():
    example_a = make_masked_example("The quick brown fox", seed=0)
    example_b = make_masked_example("Jumps over the lazy dog", seed=1)

    example_a["target_positions"] = torch.zeros((0,), dtype=torch.long)
    example_a["target_token_ids"] = torch.zeros((0,), dtype=torch.long)
    example_a["target_mask"] = torch.zeros_like(example_a["target_mask"], dtype=torch.bool)
    example_a["input_ids_ctx"] = example_a["input_ids_full"].clone()

    batch = collate_masked_examples([example_a, example_b])

    assert not torch.any(batch["target_valid_mask"][0])
    assert torch.any(batch["target_valid_mask"][1])
