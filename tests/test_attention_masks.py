import torch

from text_jepa.attention_masks import (
    block_diagonal_causal_additive_mask,
    build_packed_student_additive_attention_mask,
    causal_additive_mask,
    stack_additive_attention_masks,
)


def test_causal_additive_mask_blocks_future_attention():
    mask = causal_additive_mask(4)

    assert mask.shape == (4, 4)
    for row in range(4):
        for col in range(4):
            if col <= row:
                assert mask[row, col].item() == 0.0
            else:
                assert torch.isneginf(mask[row, col])


def test_causal_additive_mask_respects_padded_suffix():
    mask = causal_additive_mask(6, valid_length=4)

    assert mask.shape == (6, 6)
    for row in range(6):
        for col in range(6):
            if row < 4 and col < 4:
                if col <= row:
                    assert mask[row, col].item() == 0.0
                else:
                    assert torch.isneginf(mask[row, col])
            else:
                if row == col and row >= 4:
                    assert mask[row, col].item() == 0.0
                else:
                    assert torch.isneginf(mask[row, col])


def test_block_diagonal_causal_additive_mask_isolates_blocks():
    mask = block_diagonal_causal_additive_mask([2, 3])

    assert mask.shape == (5, 5)
    for row in range(5):
        for col in range(5):
            same_first_block = row < 2 and col < 2
            same_second_block = row >= 2 and col >= 2
            if same_first_block or same_second_block:
                if col <= row:
                    assert mask[row, col].item() == 0.0
                else:
                    assert torch.isneginf(mask[row, col])
            else:
                assert torch.isneginf(mask[row, col])


def test_packed_student_additive_attention_mask_stacks_full_and_packed_rows():
    mask = build_packed_student_additive_attention_mask(4, [2, 2], sequence_length=6)

    assert mask.shape == (2, 1, 6, 6)
    full_row = mask[0, 0]
    packed_row = mask[1, 0]

    for row in range(6):
        for col in range(6):
            if row < 4 and col < 4:
                if col <= row:
                    assert full_row[row, col].item() == 0.0
                else:
                    assert torch.isneginf(full_row[row, col])
            else:
                if row == col and row >= 4:
                    assert full_row[row, col].item() == 0.0
                else:
                    assert torch.isneginf(full_row[row, col])

            same_first_block = row < 2 and col < 2
            same_second_block = 2 <= row < 4 and 2 <= col < 4
            if same_first_block or same_second_block:
                if col <= row:
                    assert packed_row[row, col].item() == 0.0
                else:
                    assert torch.isneginf(packed_row[row, col])
            else:
                if row == col and row >= 4:
                    assert packed_row[row, col].item() == 0.0
                else:
                    assert torch.isneginf(packed_row[row, col])


def test_stack_additive_attention_masks_adds_batch_and_channel_dims():
    mask_a = causal_additive_mask(3)
    mask_b = block_diagonal_causal_additive_mask([1, 2])

    stacked = stack_additive_attention_masks([mask_a, mask_b])

    assert stacked.shape == (2, 1, 3, 3)
    assert torch.equal(stacked[0, 0], mask_a)
    assert torch.equal(stacked[1, 0], mask_b)
