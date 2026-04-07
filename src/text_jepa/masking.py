import re
import random

import torch

from .tokenization import load_yaml_config, tokenize_text


WORD_PATTERN = re.compile(r"\w+(?:['-]\w+)*")


def get_masking_settings(config_path):
    config = load_yaml_config(config_path)
    tokenizer_config = config.get("tokenizer") or {}
    # Masking stays separately configurable so experiments can vary corruption without changing tokenization.
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

    # The rest of the pipeline only needs scalar masking knobs, not the full nested config object.
    return max_length, float(mask_ratio), max_block_words


def find_word_spans(text):
    # Masking units are word-like spans, not punctuation characters or raw byte ranges.
    return [(match.start(), match.end()) for match in WORD_PATTERN.finditer(text)]


def map_words_to_tokens(word_spans, offset_mapping, attention_mask, special_tokens_mask):
    word_to_tokens = []

    for word_index, (word_start, word_end) in enumerate(word_spans):
        token_start = None
        token_end = None

        for token_index, ((char_start, char_end), attn, is_special) in enumerate(
            zip(offset_mapping, attention_mask, special_tokens_mask)
        ):
            # Zero-width offsets correspond to special tokens or padding in the tokenizers we support.
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
                    # Half-open token spans compose cleanly when we later merge words into contiguous blocks.
                    "token_start": token_start,
                    "token_end": token_end,
                }
            )

    return word_to_tokens


def count_maskable_tokens(attention_mask, special_tokens_mask):
    total = 0
    for attn, is_special in zip(attention_mask, special_tokens_mask):
        # The masking budget excludes pads and explicit special tokens.
        if attn == 1 and is_special == 0:
            total += 1
    return total


def sample_word_blocks(word_to_tokens, target_token_budget, max_block_words, rng):
    selected_blocks = []
    used_word_indices = set()
    masked_token_count = 0

    # Materialize all candidate blocks up front so the sampler can choose by fit quality instead of local greed.
    candidate_blocks = []
    for start_index in range(len(word_to_tokens)):
        token_count = 0
        max_end_index = min(start_index + max_block_words, len(word_to_tokens))

        for end_index in range(start_index + 1, max_end_index + 1):
            token_count += (
                word_to_tokens[end_index - 1]["token_end"] - word_to_tokens[end_index - 1]["token_start"]
            )
            candidate_blocks.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "token_count": token_count,
                }
            )

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

        # Prefer a close token-budget fit and only then break ties randomly to preserve variability.
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
            # Budget is tracked in token space because one "word" may expand to multiple subword tokens.
            masked_token_count += (
                word_to_tokens[word_index]["token_end"] - word_to_tokens[word_index]["token_start"]
            )

    return selected_blocks


def apply_mask(input_ids, target_mask, mask_token_id):
    input_ids_ctx = input_ids.clone()
    # JEPA keeps sequence length fixed and replaces only the supervised positions.
    input_ids_ctx[target_mask] = mask_token_id
    return input_ids_ctx


def extract_target_positions(input_ids_full, target_mask):
    # Keep the sparse supervision view alongside the dense mask so later modules can choose either form.
    target_positions = torch.nonzero(target_mask, as_tuple=False).squeeze(-1).to(torch.long)
    target_token_ids = input_ids_full[target_positions]
    return target_positions, target_token_ids


def build_target_mask(word_to_tokens, selected_blocks, sequence_length):
    target_mask = torch.zeros(sequence_length, dtype=torch.bool)
    masked_span_ranges_word = []
    masked_span_ranges_token = []

    for block_start, block_end in selected_blocks:
        block = word_to_tokens[block_start:block_end]
        # Adjacent words in a block collapse into one token interval so masking stays contiguous.
        token_start = min(item["token_start"] for item in block)
        token_end = max(item["token_end"] for item in block)
        word_start = min(item["word_index"] for item in block)
        word_end = max(item["word_index"] for item in block) + 1

        # Keep both coordinate systems because word-space is easier to inspect while token-space drives the model.
        target_mask[token_start:token_end] = True
        masked_span_ranges_word.append((word_start, word_end))
        masked_span_ranges_token.append((token_start, token_end))

    return target_mask, masked_span_ranges_word, masked_span_ranges_token


def mask_text(tokenizer, text, max_length, mask_ratio, max_block_words, rng=None):
    if rng is None:
        # Callers can pass a seeded RNG for deterministic tests or dataset indexing.
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

    # The ratio is approximate because masking operates on contiguous token spans, not independent tokens.
    target_token_budget = max(1, round(total_maskable_tokens * mask_ratio))
    selected_blocks = sample_word_blocks(
        word_to_tokens,
        target_token_budget,
        max_block_words,
        rng,
    )

    target_mask, masked_span_ranges_word, masked_span_ranges_token = build_target_mask(
        word_to_tokens,
        selected_blocks,
        input_ids_full.shape[0],
    )
    input_ids_ctx = apply_mask(input_ids_full, target_mask, tokenizer.mask_token_id)
    # Positions and original token ids are the predictor-facing supervision contract.
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
    # Most production call sites should depend on config-driven masking rather than threading three scalars around.
    max_length, mask_ratio, max_block_words = get_masking_settings(config_path)
    return mask_text(tokenizer, text, max_length, mask_ratio, max_block_words, rng=rng)
