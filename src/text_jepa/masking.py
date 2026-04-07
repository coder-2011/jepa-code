import re
import random

import torch

from .tokenization import load_yaml_config, tokenize_text, validate_yaml_config


WORD_PATTERN = re.compile(r"\w+(?:['-]\w+)*")


def get_masking_settings(config_path):
    config = load_yaml_config(config_path)
    validate_yaml_config(config)
    tokenizer_config = config["tokenizer"]
    # Keep masking config separate so tokenizer loading and masking can evolve independently.
    masking_config = config["masking"]

    max_length = tokenizer_config["max_length"]
    mask_ratio = masking_config.get("mask_ratio", 0.15)
    max_block_words = masking_config.get("max_block_words", 2)

    return max_length, float(mask_ratio), max_block_words


def find_word_spans(text):
    # Match word-like spans while leaving surrounding punctuation out of the masking units.
    return [(match.start(), match.end()) for match in WORD_PATTERN.finditer(text)]


def map_words_to_tokens(word_spans, offset_mapping, attention_mask, special_tokens_mask):
    word_to_tokens = []

    for word_index, (word_start, word_end) in enumerate(word_spans):
        token_start = None
        token_end = None

        for token_index, ((char_start, char_end), attn, is_special) in enumerate(
            zip(offset_mapping, attention_mask, special_tokens_mask)
        ):
            # Skip padding, special tokens, and zero-width offsets before overlap checks.
            if attn == 0 or is_special == 1 or char_start == char_end:
                continue

            if char_end <= word_start:
                continue
            if char_start >= word_end:
                break

            if token_start is None:
                token_start = token_index
            token_end = token_index + 1

        if token_start is not None:
            word_to_tokens.append(
                {
                    "word_index": word_index,
                    "char_span": (word_start, word_end),
                    # Store token spans half-open so they can be sliced directly.
                    "token_start": token_start,
                    "token_end": token_end,
                }
            )

    return word_to_tokens


def count_maskable_tokens(attention_mask, special_tokens_mask):
    total = 0
    for attn, is_special in zip(attention_mask, special_tokens_mask):
        # The mask ratio is defined over real sequence tokens, not pads or special markers.
        if attn == 1 and is_special == 0:
            total += 1
    return total


def build_candidate_blocks(word_to_tokens, max_block_words):
    candidates = []

    for start_index in range(len(word_to_tokens)):
        token_count = 0
        max_end_index = min(start_index + max_block_words, len(word_to_tokens))

        for end_index in range(start_index + 1, max_end_index + 1):
            token_count += (
                word_to_tokens[end_index - 1]["token_end"] - word_to_tokens[end_index - 1]["token_start"]
            )
            candidates.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "token_count": token_count,
                }
            )

    return candidates


def sample_word_blocks(word_to_tokens, target_token_budget, max_block_words, rng):
    selected_blocks = []
    used_word_indices = set()
    masked_token_count = 0

    # Consider all 1..max_block_words blocks so we can choose the best fit for the remaining budget.
    candidate_blocks = build_candidate_blocks(word_to_tokens, max_block_words)

    while masked_token_count < target_token_budget:
        remaining_budget = target_token_budget - masked_token_count

        available_blocks = [
            candidate
            for candidate in candidate_blocks
            if not any(
                word_index in used_word_indices
                for word_index in range(candidate["start_index"], candidate["end_index"])
            )
        ]
        if not available_blocks:
            break

        # Prefer the block that gets closest to the remaining token budget.
        # Break ties randomly so repeated runs can still explore different masks.
        best_block = min(
            available_blocks,
            key=lambda candidate: (
                abs(remaining_budget - candidate["token_count"]),
                1 if candidate["token_count"] > remaining_budget else 0,
                rng.random(),
            ),
        )

        selected_blocks.append((best_block["start_index"], best_block["end_index"]))
        for word_index in range(best_block["start_index"], best_block["end_index"]):
            used_word_indices.add(word_index)
            # Track budget in token space because subword tokenizers expand some words.
            masked_token_count += (
                word_to_tokens[word_index]["token_end"] - word_to_tokens[word_index]["token_start"]
            )

    return selected_blocks


def apply_mask(input_ids, target_mask, mask_token_id):
    input_ids_ctx = input_ids.clone()
    # Preserve sequence length and only swap the hidden positions to mask_token_id.
    input_ids_ctx[target_mask] = mask_token_id
    return input_ids_ctx


def extract_target_positions(input_ids_full, target_mask):
    # Keep both positions and original ids so later code can supervise only masked slots.
    target_positions = torch.nonzero(target_mask, as_tuple=False).squeeze(-1).to(torch.long)
    target_token_ids = input_ids_full[target_positions]
    return target_positions, target_token_ids


def mask_text(tokenizer, text, max_length, mask_ratio, max_block_words, rng=None):
    if rng is None:
        # Local RNG keeps call sites deterministic when they pass an explicit seeded Random.
        rng = random.Random()

    tokenized = tokenize_text(tokenizer, text, max_length)
    input_ids_full = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    offset_mapping = tokenized["offset_mapping"]
    special_tokens_mask = tokenized["special_tokens_mask"]

    word_spans = find_word_spans(text)
    word_to_tokens = map_words_to_tokens(
        word_spans,
        offset_mapping,
        attention_mask.tolist(),
        special_tokens_mask,
    )
    if not word_to_tokens:
        raise ValueError("No maskable words were found in the input text")

    total_maskable_tokens = count_maskable_tokens(attention_mask.tolist(), special_tokens_mask)
    if total_maskable_tokens <= 0:
        raise ValueError("No maskable tokens were found in the tokenized example")

    # The ratio is still approximate, but we now try to choose blocks that fit the token budget closely.
    target_token_budget = max(1, round(total_maskable_tokens * mask_ratio))
    selected_blocks = sample_word_blocks(
        word_to_tokens,
        target_token_budget,
        max_block_words,
        rng,
    )

    target_mask = torch.zeros(input_ids_full.shape[0], dtype=torch.bool)
    masked_span_ranges_word = []
    masked_span_ranges_token = []

    for block_start, block_end in selected_blocks:
        block = word_to_tokens[block_start:block_end]
        # Collapse each chosen word block into one contiguous token span for masking.
        token_start = min(item["token_start"] for item in block)
        token_end = max(item["token_end"] for item in block)
        word_start = min(item["word_index"] for item in block)
        word_end = max(item["word_index"] for item in block) + 1

        # Keep both word-space and token-space ranges for debugging and future inspections.
        target_mask[token_start:token_end] = True
        masked_span_ranges_word.append((word_start, word_end))
        masked_span_ranges_token.append((token_start, token_end))

    input_ids_ctx = apply_mask(input_ids_full, target_mask, tokenizer.mask_token_id)
    target_positions, target_token_ids = extract_target_positions(input_ids_full, target_mask)

    return {
        "input_ids_full": input_ids_full,
        "input_ids_ctx": input_ids_ctx,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
        "target_positions": target_positions,
        "target_token_ids": target_token_ids,
        "masked_span_ranges_word": masked_span_ranges_word,
        "masked_span_ranges_token": masked_span_ranges_token,
    }


def mask_text_from_yaml(tokenizer, text, config_path, rng=None):
    # This is the convenience entry point most callers should use.
    max_length, mask_ratio, max_block_words = get_masking_settings(config_path)
    return mask_text(tokenizer, text, max_length, mask_ratio, max_block_words, rng=rng)
