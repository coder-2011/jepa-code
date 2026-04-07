from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer


def load_yaml_config(config_path):
    path = Path(config_path)
    # Default to an empty dict so missing sections fail through our own validation.
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_yaml_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    if "tokenizer" not in config:
        raise ValueError("Missing 'tokenizer' section in the YAML config")
    if "masking" not in config:
        raise ValueError("Missing 'masking' section in the YAML config")

    tokenizer_config = config["tokenizer"]
    masking_config = config["masking"]

    if not isinstance(tokenizer_config, dict):
        raise ValueError("'tokenizer' must be a YAML mapping")
    if not isinstance(masking_config, dict):
        raise ValueError("'masking' must be a YAML mapping")

    model_name = tokenizer_config.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("'tokenizer.model_name' must be a non-empty string")

    max_length = tokenizer_config.get("max_length")
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("'tokenizer.max_length' must be a positive integer")

    mask_token = tokenizer_config.get("mask_token", "[MASK]")
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError("'tokenizer.mask_token' must be a non-empty string when provided")

    mask_ratio = masking_config.get("mask_ratio", 0.15)
    if not isinstance(mask_ratio, (int, float)) or not 0 < mask_ratio < 1:
        raise ValueError("'masking.mask_ratio' must be between 0 and 1")

    max_block_words = masking_config.get("max_block_words", 2)
    if not isinstance(max_block_words, int) or max_block_words <= 0:
        raise ValueError("'masking.max_block_words' must be a positive integer")


def load_tokenizer_from_yaml(config_path):
    config = load_yaml_config(config_path)
    validate_yaml_config(config)
    # Read only the tokenizer section here; masking settings live in masking.py.
    tokenizer_config = config["tokenizer"]

    model_name = tokenizer_config.get("model_name")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # The masker depends on character offsets, which are only available on fast tokenizers.
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required because masking uses offset mappings")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer needs either a pad token or an eos token")
        # Reuse eos as pad so fixed-length batching works even for causal-LM tokenizers.
        tokenizer.pad_token = tokenizer.eos_token

    desired_mask_token = tokenizer_config.get("mask_token", "[MASK]")
    if tokenizer.mask_token is None:
        # Qwen-style tokenizers may not ship with a JEPA-ready mask token.
        tokenizer.add_special_tokens({"mask_token": desired_mask_token})

    return tokenizer


def tokenize_text(tokenizer, text, max_length):
    # Return offsets and special-token markers because the masker operates in word space,
    # then projects those word spans back onto token positions.
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
        # Tensors keep the training-facing pieces ready for downstream batching code.
        "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        # Leave offsets and special-token markers as Python lists; masking uses simple iteration.
        "offset_mapping": encoded["offset_mapping"],
        "special_tokens_mask": encoded["special_tokens_mask"],
    }


def get_tokenizer_metadata(tokenizer):
    # Keep this minimal: only ids the rest of the pipeline is likely to care about.
    return {
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "vocab_size": len(tokenizer),
    }
