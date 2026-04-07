from pathlib import Path
from transformers import AutoTokenizer
import torch
import yaml


def load_yaml_config(config_path):
    path = Path(config_path)
    # Config parsing stays centralized so tokenizer and masking loaders agree on the same file semantics.
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_tokenizer_from_yaml(config_path):
    config = load_yaml_config(config_path)
    # Tokenizer setup intentionally reads only its own section; masking settings are consumed elsewhere.
    tokenizer_config = config.get("tokenizer") or {}

    model_name = tokenizer_config.get("model_name")

    max_length = tokenizer_config.get("max_length")
    # Max length is validated here because batching and position embeddings depend on it being trustworthy.
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("tokenizer.max_length must be a positive integer")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Reuse eos as pad so fixed-length batching still works for tokenizers that ship without one.
    tokenizer.pad_token = tokenizer.eos_token

    desired_mask_token = tokenizer_config.get("mask_token", "[MASK]")
    if tokenizer.mask_token is None:
        # Some causal-LM tokenizers do not include a dedicated mask token.
        tokenizer.add_special_tokens({"mask_token": desired_mask_token})

    return tokenizer


def tokenize_text(tokenizer, text, max_length):
    # The tokenizer returns enough metadata for span masking without forcing later code back into HF internals.
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )

    return {
        # IDs and attention masks become tensors immediately because the training path consumes them directly.
        "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        # The masker walks character offsets and special-token flags directly.
        "offset_mapping": encoded["offset_mapping"],
        "special_tokens_mask": encoded["special_tokens_mask"],
    }


def get_tokenizer_metadata(tokenizer):
    # Keep this boundary small so callers do not start depending on the full tokenizer object shape.
    return {
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "vocab_size": len(tokenizer),
    }
