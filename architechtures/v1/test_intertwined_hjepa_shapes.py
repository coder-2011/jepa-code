from dataclasses import replace

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from intertwined_hjepa import IntertwinedBlock, IntertwinedConfig, IntertwinedHJEPA
from text_helpers import HFTokenizer, LMHead, TokenEmbeddings

with open("intertwined_hjepa.yaml", "r", encoding="utf-8") as handle:
    YAML_CONFIG = IntertwinedConfig(**yaml.safe_load(handle))


def make_config():
    return replace(
        YAML_CONFIG,
        vocab_size=32,
        max_length=8,
        residual_dim=8,
        compressed_dim=4,
        depth=3,
        num_heads=2,
        predictor_hidden_dim=16,
        dropout=0.0,
        ema_momentum=0.5,
        jepa_warmup_steps=0,
    )


def test_block_student_forward_shapes():
    block = IntertwinedBlock(
        residual_dim=8,
        compressed_dim=4,
        predictor_hidden_dim=16,
        num_heads=2,
        dropout=0.0,
    )
    out = block.forward_student(torch.randn(2, 5, 8))

    assert out["x_next"].shape == (2, 5, 8)
    assert out["x_post_attn"].shape == (2, 5, 8)
    assert out["z"].shape == (2, 5, 4)
    assert out["delta"].shape == (2, 5, 4)


def test_text_helpers_shapes():
    embeddings = TokenEmbeddings(vocab_size=32, max_length=8, residual_dim=8)
    hidden_states = embeddings(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long))
    tied_head = LMHead(8, 32, embeddings.token_embedding, tie_weights=True)
    untied_head = LMHead(8, 32, embeddings.token_embedding, tie_weights=False)

    assert hidden_states.shape == (2, 4, 8)
    assert tied_head(hidden_states).shape == (2, 4, 32)
    assert untied_head(hidden_states).shape == (2, 4, 32)


def test_hf_tokenizer_wrapper_offline():
    tokenizer = Tokenizer(WordLevel({"<pad>": 0, "hello": 1, "world": 2}, unk_token="<pad>"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = HFTokenizer(
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="<pad>",
            unk_token="<pad>",
        )
    )

    assert wrapped.vocab_size == 3
    assert wrapped.encode("hello world", add_special_tokens=False) == [1, 2]
    assert wrapped.decode([1, 2]) == "hello world"


def test_config_loads_from_yaml_and_builds_model():
    config = YAML_CONFIG
    model = IntertwinedHJEPA(config)

    assert config.vocab_size == 256
    assert config.max_length == 128
    assert config.residual_dim == 256
    assert config.compressed_dim == 128
    assert len(model.blocks) == config.depth
    assert model.embeddings.token_embedding.num_embeddings == config.vocab_size


def test_model_forward_returns_expected_shapes():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs["logits"].shape == (2, 4, 32)
    assert outputs["final_states"].shape == (2, 4, 8)
    assert outputs["loss"].ndim == 0
    assert outputs["loss_main"].ndim == 0
    assert outputs["loss_jepa"].ndim == 0
    assert len(outputs["z"]) == 3
    assert len(outputs["deltas"]) == 3
    assert len(outputs["targets"]) == 2


def test_model_forward_without_labels_uses_jepa_loss():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    outputs = model(input_ids=input_ids)

    assert outputs["loss_main"] is None
    assert torch.equal(outputs["loss"], make_config().lambda_jepa * outputs["loss_jepa"])


def test_depth_must_allow_future_layer_target():
    config = replace(make_config(), depth=1)

    try:
        IntertwinedHJEPA(config)
    except AssertionError as exc:
        assert "depth" in str(exc)
    else:
        raise AssertionError("Expected depth < 2 to be rejected")
