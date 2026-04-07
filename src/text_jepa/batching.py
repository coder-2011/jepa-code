import torch


REQUIRED_BATCH_KEYS = {
    "input_ids_full",
    "input_ids_ctx",
    "attention_mask",
    "target_mask",
    "target_positions",
    "target_token_ids",
}


def validate_and_collect_examples(examples):
    if not examples:
        raise ValueError("examples must be non-empty")

    # Sequence length is fixed by tokenization, so any mismatch here means the data pipeline drifted.
    sequence_length = examples[0]["input_ids_full"].shape[0]
    input_ids_full_list = []
    input_ids_ctx_list = []
    attention_mask_list = []
    target_mask_list = []
    target_positions_list = []
    target_token_ids_list = []
    t_max = 0

    for example in examples:
        missing = REQUIRED_BATCH_KEYS - example.keys()
        if missing:
            raise ValueError(f"example is missing keys: {sorted(missing)}")

        input_ids_full = example["input_ids_full"]
        input_ids_ctx = example["input_ids_ctx"]
        attention_mask = example["attention_mask"]
        target_mask = example["target_mask"]
        target_positions = example["target_positions"]
        target_token_ids = example["target_token_ids"]

        if input_ids_full.shape[0] != sequence_length:
            raise ValueError("all examples must share the same sequence length")
        if input_ids_ctx.shape[0] != sequence_length:
            raise ValueError("input_ids_ctx must match input_ids_full length")
        if attention_mask.shape[0] != sequence_length:
            raise ValueError("attention_mask must match input_ids_full length")
        if target_mask.shape[0] != sequence_length:
            raise ValueError("target_mask must match input_ids_full length")
        # Predictor-side tensors must stay aligned sample by sample before we batch-pad them.
        if target_positions.shape[0] != target_token_ids.shape[0]:
            raise ValueError("target_positions and target_token_ids must have the same length")

        input_ids_full_list.append(input_ids_full)
        input_ids_ctx_list.append(input_ids_ctx)
        attention_mask_list.append(attention_mask)
        target_mask_list.append(target_mask)
        target_positions_list.append(target_positions)
        target_token_ids_list.append(target_token_ids)
        t_max = max(t_max, target_positions.shape[0])

    return {
        "input_ids_full_list": input_ids_full_list,
        "input_ids_ctx_list": input_ids_ctx_list,
        "attention_mask_list": attention_mask_list,
        "target_mask_list": target_mask_list,
        "target_positions_list": target_positions_list,
        "target_token_ids_list": target_token_ids_list,
        "t_max": t_max,
    }


def collate_masked_examples(examples):
    collected = validate_and_collect_examples(examples)
    input_ids_full_list = collected["input_ids_full_list"]
    input_ids_ctx_list = collected["input_ids_ctx_list"]
    attention_mask_list = collected["attention_mask_list"]
    target_mask_list = collected["target_mask_list"]
    target_positions_list = collected["target_positions_list"]
    target_token_ids_list = collected["target_token_ids_list"]
    t_max = collected["t_max"]

    # These tensors already share shape (L,), so batching is just a stack.
    input_ids_full = torch.stack(input_ids_full_list, dim=0)
    input_ids_ctx = torch.stack(input_ids_ctx_list, dim=0)
    attention_mask = torch.stack(attention_mask_list, dim=0)
    target_mask = torch.stack(target_mask_list, dim=0)

    batch_size = len(examples)

    # Zero padding is fine here because target_valid_mask tells downstream code which slots are real.
    target_positions = torch.zeros((batch_size, t_max), dtype=torch.long)
    target_token_ids = torch.zeros((batch_size, t_max), dtype=torch.long)
    target_valid_mask = torch.zeros((batch_size, t_max), dtype=torch.bool)

    for batch_index, (target_positions_example, target_token_ids_example) in enumerate(
        zip(target_positions_list, target_token_ids_list)
    ):
        target_count = target_positions_example.shape[0]
        # Left-pack valid targets so masking logic downstream can ignore only the padded suffix.
        target_positions[batch_index, :target_count] = target_positions_example
        target_token_ids[batch_index, :target_count] = target_token_ids_example
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
