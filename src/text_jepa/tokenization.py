from pathlib import Path
from transformers import AutoTokenizer
import torch
import yaml


def load_yaml_config(config_path):
    path = Path(config_path)
    # Default to an empty dict so missing sections fail through our own validation.
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_tokenizer_from_yaml(config_path):
    if AutoTokenizer is None:
        raise ImportError("transformers is required to load the Hugging Face tokenizer")

    config = load_yaml_config(config_path)
    # Read only the tokenizer section here; masking settings live in masking.py.
    tokenizer_config = config.get("tokenizer") or {}

    model_name = tokenizer_config.get("model_name")
    if not model_name:
        raise ValueError("tokenizer.model_name must be set in the YAML config")

    max_length = tokenizer_config.get("max_length")
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("tokenizer.max_length must be a positive integer")

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
