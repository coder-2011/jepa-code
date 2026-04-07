import random

import torch

from .tokenization import load_yaml_config, tokenize_text


def get_masking_settings(config_path):
    config = load_yaml_config(config_path)
    tokenizer_config = config.get("tokenizer") or {}
    masking_config = config.get("masking") or {}

    max_length = tokenizer_config.get("max_length")
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("tokenizer.max_length must be a positive integer")

    mask_ratio = masking_config.get("mask_ratio", 0.15)
    if not isinstance(mask_ratio, (int, float)) or not 0 < mask_ratio < 1:
        raise ValueError("masking.mask_ratio must be between 0 and 1")

    max_block_words = masking_config.get("max_block_words", 2)
    if not isinstance(max_block_words, int) or max_block_words <= 0:
        raise ValueError("masking.max_block_words must be a positive integer")

    return max_length, float(mask_ratio), max_block_words


def find_word_spans(text):
    spans = []
    in_word = False
    start = None

    for index, char in enumerate(text):
        if not char.isspace() and not in_word:
            start = index
            in_word = True
        elif char.isspace() and in_word:
            spans.append((start, index))
            in_word = False

    if in_word:
        spans.append((start, len(text)))

    return spans


def map_words_to_tokens(word_spans, offset_mapping, attention_mask, special_tokens_mask):
    word_to_tokens = []

    for word_index, (word_start, word_end) in enumerate(word_spans):
        token_indices = []

        for token_index, ((token_start, token_end), attn, is_special) in enumerate(
            zip(offset_mapping, attention_mask, special_tokens_mask)
        ):
            if attn == 0 or is_special == 1 or token_start == token_end:
                continue

            overlaps = not (token_end <= word_start or token_start >= word_end)
            if overlaps:
                token_indices.append(token_index)

        if token_indices:
            word_to_tokens.append(
                {
                    "word_index": word_index,
                    "char_span": (word_start, word_end),
                    "token_start": min(token_indices),
                    "token_end": max(token_indices) + 1,
                }
            )

    return word_to_tokens


def count_maskable_tokens(attention_mask, special_tokens_mask):
    total = 0
    for attn, is_special in zip(attention_mask, special_tokens_mask):
        if attn == 1 and is_special == 0:
            total += 1
    return total


def sample_word_blocks(word_to_tokens, target_token_budget, max_block_words, rng):
    selected_blocks = []
    used_word_indices = set()
    masked_token_count = 0

    candidate_starts = list(range(len(word_to_tokens)))
    rng.shuffle(candidate_starts)

    for start_index in candidate_starts:
        if masked_token_count >= target_token_budget:
            break
        if start_index in used_word_indices:
            continue

        block_size = rng.randint(1, max_block_words)
        end_index = min(start_index + block_size, len(word_to_tokens))

        if any(word_index in used_word_indices for word_index in range(start_index, end_index)):
            continue

        selected_blocks.append((start_index, end_index))
        for word_index in range(start_index, end_index):
            used_word_indices.add(word_index)
            masked_token_count += (
                word_to_tokens[word_index]["token_end"] - word_to_tokens[word_index]["token_start"]
            )

    return selected_blocks


def apply_mask(input_ids, target_mask, mask_token_id):
    input_ids_ctx = input_ids.clone()
    input_ids_ctx[target_mask] = mask_token_id
    return input_ids_ctx


def extract_target_positions(input_ids_full, target_mask):
    target_positions = torch.nonzero(target_mask, as_tuple=False).squeeze(-1).to(torch.long)
    target_token_ids = input_ids_full[target_positions]
    return target_positions, target_token_ids


def mask_text(tokenizer, text, max_length, mask_ratio, max_block_words, rng=None):
    if rng is None:
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
        token_start = min(item["token_start"] for item in block)
        token_end = max(item["token_end"] for item in block)
        word_start = min(item["word_index"] for item in block)
        word_end = max(item["word_index"] for item in block) + 1

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
    max_length, mask_ratio, max_block_words = get_masking_settings(config_path)
    return mask_text(tokenizer, text, max_length, mask_ratio, max_block_words, rng=rng)
