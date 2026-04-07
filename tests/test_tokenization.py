from text_jepa.tokenization import (
    get_tokenizer_metadata,
    load_tokenizer_from_yaml,
    tokenize_text,
)

from conftest import FakeTokenizer, write_test_config


def test_loads_qwen_tokenizer_from_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_test_config(config_path)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, use_fast=True):
            assert model_name == "Qwen/Qwen3-0.6B"
            assert use_fast is True
            return FakeTokenizer(has_mask_token=False)

    # Patch the loading boundary we own so the test stays offline and deterministic.
    monkeypatch.setattr("text_jepa.tokenization.AutoTokenizer", FakeAutoTokenizer)

    tokenizer = load_tokenizer_from_yaml(config_path)

    # The loader should synthesize a usable mask token when the pretrained tokenizer lacks one.
    assert tokenizer.mask_token == "[MASK]"
    assert tokenizer.mask_token_id == 99


def test_tokenize_text_returns_required_fields():
    tokenizer = FakeTokenizer()

    encoded = tokenize_text(tokenizer, "The quick brown fox", max_length=8)

    # The masking code depends on this exact field set.
    assert set(encoded) == {
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "special_tokens_mask",
    }
    assert encoded["input_ids"].shape[0] == 8
    assert encoded["attention_mask"].shape[0] == 8
    assert len(encoded["offset_mapping"]) == 8
    assert len(encoded["special_tokens_mask"]) == 8


def test_get_tokenizer_metadata_returns_special_ids():
    tokenizer = FakeTokenizer()

    metadata = get_tokenizer_metadata(tokenizer)

    # Metadata should stay intentionally small and stable.
    assert metadata["pad_token_id"] == 0
    assert metadata["mask_token_id"] == 99
    assert metadata["bos_token_id"] == 2
    assert metadata["eos_token_id"] == 1
    assert metadata["vocab_size"] == 128
