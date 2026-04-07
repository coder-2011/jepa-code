import torch


REQUIRED_BATCH_KEYS = {
    "input_ids_full",
    "input_ids_ctx",
    "attention_mask",
    "target_mask",
    "target_positions",
    "target_token_ids",
}


def validate_example_batch(examples):
    if not examples:
        raise ValueError("examples must be non-empty")

    # Sequence length is fixed by tokenization, so any mismatch here means the data pipeline drifted.
    sequence_length = examples[0]["input_ids_full"].shape[0]

    for example in examples:
        missing = REQUIRED_BATCH_KEYS - example.keys()
        if missing:
            raise ValueError(f"example is missing keys: {sorted(missing)}")

        if example["input_ids_full"].shape[0] != sequence_length:
            raise ValueError("all examples must share the same sequence length")
        if example["input_ids_ctx"].shape[0] != sequence_length:
            raise ValueError("input_ids_ctx must match input_ids_full length")
        if example["attention_mask"].shape[0] != sequence_length:
            raise ValueError("attention_mask must match input_ids_full length")
        if example["target_mask"].shape[0] != sequence_length:
            raise ValueError("target_mask must match input_ids_full length")
        # Predictor-side tensors must stay aligned sample by sample before we batch-pad them.
        if example["target_positions"].shape[0] != example["target_token_ids"].shape[0]:
            raise ValueError("target_positions and target_token_ids must have the same length")


def collate_masked_examples(examples):
    validate_example_batch(examples)

    # These tensors already share shape (L,), so batching is just a stack.
    input_ids_full = torch.stack([example["input_ids_full"] for example in examples], dim=0)
    input_ids_ctx = torch.stack([example["input_ids_ctx"] for example in examples], dim=0)
    attention_mask = torch.stack([example["attention_mask"] for example in examples], dim=0)
    target_mask = torch.stack([example["target_mask"] for example in examples], dim=0)

    # Only the target-side tensors are ragged; pad them to the batch maximum target count.
    t_max = max(example["target_positions"].shape[0] for example in examples)
    batch_size = len(examples)

    # Zero padding is fine here because target_valid_mask tells downstream code which slots are real.
    target_positions = torch.zeros((batch_size, t_max), dtype=torch.long)
    target_token_ids = torch.zeros((batch_size, t_max), dtype=torch.long)
    target_valid_mask = torch.zeros((batch_size, t_max), dtype=torch.bool)

    for batch_index, example in enumerate(examples):
        target_count = example["target_positions"].shape[0]
        # Left-pack valid targets so masking logic downstream can ignore only the padded suffix.
        target_positions[batch_index, :target_count] = example["target_positions"]
        target_token_ids[batch_index, :target_count] = example["target_token_ids"]
        target_valid_mask[batch_index, :target_count] = True

    return {
        "input_ids_full": input_ids_full,
        "input_ids_ctx": input_ids_ctx,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
        "target_positions": target_positions,
        "target_token_ids": target_token_ids,
        "target_valid_mask": target_valid_mask,
    }
