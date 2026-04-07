# Batching PRD

## Status

Draft for the next text-JEPA implementation milestone.

## Goal

Take a list of masked single-example outputs and turn them into one model-ready batch.

This layer sits between:

1. tokenizer + masker
2. JEPA model

## Short Answer

Yes, stacking is the right idea.

For this repo, batching should stay simple:

- stack the fixed-length sequence tensors
- pad the variable-length target tensors
- return one clean batch dict

That is all we need for v1.

## Why Stacking Is Correct

After tokenization, every example already has the same sequence length `L`.

So these fields should just be stacked:

- `input_ids_full`
- `input_ids_ctx`
- `attention_mask`
- `target_mask`

This is simple and correct because those tensors are already shape-aligned per example.

The only thing that is still variable per example is the number of target tokens:

- `target_positions`
- `target_token_ids`

Those need padding to `T_max`.

## Input Contract

Each example coming from the masker should provide:

- `input_ids_full: (L,)`
- `input_ids_ctx: (L,)`
- `attention_mask: (L,)`
- `target_mask: (L,)`
- `target_positions: (T_i,)`
- `target_token_ids: (T_i,)`

## Output Contract

The collator should return:

- `input_ids_full: (B, L)`
- `input_ids_ctx: (B, L)`
- `attention_mask: (B, L)`
- `target_mask: (B, L)`
- `target_positions: (B, T_max)`
- `target_token_ids: (B, T_max)`
- `target_valid_mask: (B, T_max)`

Where:

- `B` is batch size
- `L` is fixed sequence length
- `T_max` is the largest target count in the batch

## What the Collator Does

1. Check that the batch is non-empty.
2. Check that all examples have the required keys.
3. Check that all sequence tensors have the same length `L`.
4. Stack the fixed-length sequence tensors.
5. Find `T_max`.
6. Create padded target tensors of shape `(B, T_max)`.
7. Fill valid target slots from each example.
8. Mark those slots in `target_valid_mask`.

## Pseudocode

```python
def collate_masked_examples(examples):
    if not examples:
        raise ValueError("examples must be non-empty")

    required_keys = {
        "input_ids_full",
        "input_ids_ctx",
        "attention_mask",
        "target_mask",
        "target_positions",
        "target_token_ids",
    }

    sequence_length = examples[0]["input_ids_full"].shape[0]

    for example in examples:
        missing = required_keys - example.keys()
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
        if example["target_positions"].shape[0] != example["target_token_ids"].shape[0]:
            raise ValueError("target_positions and target_token_ids must have the same length")

    input_ids_full = torch.stack([ex["input_ids_full"] for ex in examples], dim=0)
    input_ids_ctx = torch.stack([ex["input_ids_ctx"] for ex in examples], dim=0)
    attention_mask = torch.stack([ex["attention_mask"] for ex in examples], dim=0)
    target_mask = torch.stack([ex["target_mask"] for ex in examples], dim=0)

    t_max = max(ex["target_positions"].shape[0] for ex in examples)
    batch_size = len(examples)

    target_positions = torch.zeros((batch_size, t_max), dtype=torch.long)
    target_token_ids = torch.zeros((batch_size, t_max), dtype=torch.long)
    target_valid_mask = torch.zeros((batch_size, t_max), dtype=torch.bool)

    for batch_index, example in enumerate(examples):
        t = example["target_positions"].shape[0]
        target_positions[batch_index, :t] = example["target_positions"]
        target_token_ids[batch_index, :t] = example["target_token_ids"]
        target_valid_mask[batch_index, :t] = True

    return {
        "input_ids_full": input_ids_full,
        "input_ids_ctx": input_ids_ctx,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
        "target_positions": target_positions,
        "target_token_ids": target_token_ids,
        "target_valid_mask": target_valid_mask,
    }
```

## Example

If one example has:

- `target_positions.shape = (3,)`

and another has:

- `target_positions.shape = (5,)`

then:

- `T_max = 5`

and the collator returns:

- `target_positions.shape = (2, 5)`
- `target_token_ids.shape = (2, 5)`
- `target_valid_mask.shape = (2, 5)`

with:

- first row valid mask: `[True, True, True, False, False]`
- second row valid mask: `[True, True, True, True, True]`

## Important Notes

### 1. Why do we need `target_valid_mask`?

Because padding `target_positions` with zero is ambiguous.

Position `0` is a real sequence position, so the model and loss must use `target_valid_mask` to know which target slots are real.

### 2. Should we collate span-range metadata?

Not in v1.

Fields like:

- `masked_span_ranges_word`
- `masked_span_ranges_token`

can stay out of the collated batch for now.

### 3. Should we support empty-target examples?

Yes, mechanically.

If an example has zero target positions:

- its row in `target_valid_mask` should be all `False`

That keeps the collator simple and makes the behavior explicit.

## Test Plan

Create `tests/test_batching.py`.

Minimum tests:

1. stacks `input_ids_full`, `input_ids_ctx`, `attention_mask`, and `target_mask`
2. pads `target_positions` and `target_token_ids` to `T_max`
3. builds `target_valid_mask` correctly
4. preserves valid target values
5. rejects empty input
6. rejects missing required keys
7. rejects sequence-length mismatch
8. supports an example with zero target tokens

## Recommended Implementation

Keep this file small.

Recommended shape:

- `src/text_jepa/batching.py`
- one main function:
  - `collate_masked_examples(examples)`

If a helper is needed, only add one if it makes the main function noticeably clearer.

## Final Recommendation

Do batching with:

- stacking for fixed-length tensors
- padding for variable-length target tensors

That is the cleanest next step and enough for the first JEPA model pass.
