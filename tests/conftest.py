import re
import sys
from pathlib import Path

import yaml

# Keep tests importable from the repo root without depending on an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write_test_config(
    path,
    model_name="Qwen/Qwen3-0.6B",
    seed=0,
    deterministic=True,
    max_length=12,
    mask_ratio=0.15,
    max_block_words=2,
    hidden_dim=8,
    num_heads=2,
    num_layers=2,
    ffn_dim=32,
    dropout=0.0,
    norm="rms",
    ema_momentum=0.996,
):
    # One shared writer keeps every config-driven test on the same YAML contract.
    path.write_text(
        yaml.safe_dump(
            {
                "tokenizer": {
                    "model_name": model_name,
                    "max_length": max_length,
                    "mask_token": "[MASK]",
                },
                "runtime": {
                    "seed": seed,
                    "deterministic": deterministic,
                },
                "masking": {
                    "mask_ratio": mask_ratio,
                    "max_block_words": max_block_words,
                },
                "model": {
                    "hidden_dim": hidden_dim,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "ffn_dim": ffn_dim,
                    "dropout": dropout,
                    "norm": norm,
                    "ema_momentum": ema_momentum,
                },
            }
        ),
        encoding="utf-8",
    )


class FakeTokenizer:
    def __init__(self, has_mask_token=True):
        # Mirror only the HF attributes our code actually reads so tests stay lightweight.
        self.is_fast = True
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.mask_token = "[MASK]" if has_mask_token else None
        self.mask_token_id = 99 if has_mask_token else None

    def add_special_tokens(self, mapping):
        # Simulate the side effect that matters to the repo: a new mask token id becomes available.
        self.mask_token = mapping["mask_token"]
        self.mask_token_id = 99
        return 1

    def __len__(self):
        return 128

    def __call__(
        self,
        text,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    ):
        if isinstance(text, list):
            raise NotImplementedError("FakeTokenizer only supports single examples in tests")

        # Start with an explicit BOS-like special token so special-token exclusion gets exercised.
        input_ids = [self.bos_token_id]
        attention_mask = [1]
        offset_mapping = [(0, 0)]
        special_tokens_mask = [1]

        for index, match in enumerate(re.finditer(r"\S+", text)):
            # Give each visible surface token one learned id so word-to-token mapping stays easy to inspect.
            input_ids.append(10 + index)
            attention_mask.append(1)
            offset_mapping.append((match.start(), match.end()))
            special_tokens_mask.append(0)

        # Add EOS so the tokenizer output includes both leading and trailing special tokens.
        input_ids.append(self.eos_token_id)
        attention_mask.append(1)
        offset_mapping.append((0, 0))
        special_tokens_mask.append(1)

        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        offset_mapping = offset_mapping[:max_length]
        special_tokens_mask = special_tokens_mask[:max_length]

        # Zero-width offsets make padding impossible to confuse with maskable text tokens.
        while len(input_ids) < max_length:
            input_ids.append(self.pad_token_id)
            attention_mask.append(0)
            offset_mapping.append((0, 0))
            special_tokens_mask.append(1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "special_tokens_mask": special_tokens_mask,
        }
