import torch


def _normalize_lengths(lengths):
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    lengths = tuple(int(length) for length in lengths)
    if not lengths:
        raise ValueError("lengths must be non-empty")
    if any(length <= 0 for length in lengths):
        raise ValueError("all lengths must be positive integers")
    return lengths


def causal_additive_mask(sequence_length, *, valid_length=None, device=None, dtype=torch.float32):
    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if valid_length is None:
        valid_length = sequence_length
    valid_length = int(valid_length)
    if valid_length <= 0:
        raise ValueError("valid_length must be positive")
    if valid_length > sequence_length:
        raise ValueError("valid_length cannot exceed sequence_length")

    mask = torch.full((sequence_length, sequence_length), float("-inf"), device=device, dtype=dtype)
    visible = torch.zeros((valid_length, valid_length), device=device, dtype=dtype)
    future_positions = torch.triu(
        torch.ones((valid_length, valid_length), device=device, dtype=torch.bool),
        diagonal=1,
    )
    visible = visible.masked_fill(future_positions, float("-inf"))
    mask[:valid_length, :valid_length] = visible
    if valid_length < sequence_length:
        padding_rows = torch.arange(valid_length, sequence_length, device=device)
        mask[padding_rows, padding_rows] = 0.0
    return mask


def block_diagonal_causal_additive_mask(block_lengths, *, sequence_length=None, device=None, dtype=torch.float32):
    block_lengths = _normalize_lengths(block_lengths)
    visible_length = sum(block_lengths)

    if sequence_length is None:
        sequence_length = visible_length
    sequence_length = int(sequence_length)
    if sequence_length < visible_length:
        raise ValueError("sequence_length cannot be smaller than the sum of block_lengths")

    mask = torch.full((sequence_length, sequence_length), float("-inf"), device=device, dtype=dtype)
    start = 0
    for block_length in block_lengths:
        end = start + block_length
        mask[start:end, start:end] = causal_additive_mask(
            block_length,
            device=device,
            dtype=dtype,
        )
        start = end
    if visible_length < sequence_length:
        padding_rows = torch.arange(visible_length, sequence_length, device=device)
        mask[padding_rows, padding_rows] = 0.0
    return mask


def stack_additive_attention_masks(masks, *, device=None, dtype=None):
    masks = tuple(masks)
    if not masks:
        raise ValueError("masks must be non-empty")

    normalized = []
    expected_shape = None
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            tensor = mask.to(device=device, dtype=dtype) if device is not None or dtype is not None else mask
        else:
            tensor = torch.tensor(mask, device=device, dtype=dtype)
        if tensor.ndim != 2:
            raise ValueError("each mask must have shape (L, L)")
        if expected_shape is None:
            expected_shape = tensor.shape
        elif tensor.shape != expected_shape:
            raise ValueError("all masks must have the same shape")
        normalized.append(tensor)

    return torch.stack(normalized, dim=0).unsqueeze(1)


def build_packed_student_additive_attention_mask(
    full_sequence_length,
    packed_block_lengths,
    *,
    sequence_length=None,
    device=None,
    dtype=torch.float32,
):
    packed_block_lengths = _normalize_lengths(packed_block_lengths)
    if sequence_length is None:
        sequence_length = max(int(full_sequence_length), sum(packed_block_lengths))
    full_mask = causal_additive_mask(
        sequence_length,
        valid_length=full_sequence_length,
        device=device,
        dtype=dtype,
    )
    packed_mask = block_diagonal_causal_additive_mask(
        packed_block_lengths,
        sequence_length=sequence_length,
        device=device,
        dtype=dtype,
    )
    return stack_additive_attention_masks([full_mask, packed_mask], device=device, dtype=dtype)
