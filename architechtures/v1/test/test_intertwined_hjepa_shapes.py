from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from intertwined_hjepa import (
    CausalSelfAttention,
    FinalResidualBlock,
    IntertwinedBlock,
    IntertwinedConfig,
    IntertwinedHJEPA,
)
from text_helpers import HFTokenizer, LMHead, TokenEmbeddings

ROOT = Path(__file__).resolve().parent.parent

YAML_CONFIG = IntertwinedConfig.from_yaml(ROOT / "intertwined_hjepa.yaml")


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
        beta_sigreg=0.0,
        sigreg_warmup_steps=0,
    )


def test_block_student_forward_shapes():
    config = make_config()
    block = IntertwinedBlock(
        residual_dim=config.residual_dim,
        compressed_dim=config.compressed_dim,
        predictor_hidden_dim=config.predictor_hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )
    batch_size = 2
    sequence_length = min(5, config.max_length)
    out = block.forward_student(torch.randn(batch_size, sequence_length, config.residual_dim))

    assert out["x_next"].shape == (batch_size, sequence_length, config.residual_dim)
    assert out["x_post_attn"].shape == (batch_size, sequence_length, config.residual_dim)
    assert out["z"].shape == (batch_size, sequence_length, config.compressed_dim)
    assert out["delta"].shape == (batch_size, sequence_length, config.compressed_dim)


def test_final_residual_block_shapes():
    config = make_config()
    block = FinalResidualBlock(
        residual_dim=config.residual_dim,
        hidden_dim=config.predictor_hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )
    batch_size = 2
    sequence_length = min(5, config.max_length)
    out = block(torch.randn(batch_size, sequence_length, config.residual_dim))

    assert out["x_next"].shape == (batch_size, sequence_length, config.residual_dim)
    assert out["x_post_attn"].shape == (batch_size, sequence_length, config.residual_dim)


def test_text_helpers_shapes():
    config = make_config()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    embeddings = TokenEmbeddings(
        vocab_size=config.vocab_size,
        max_length=config.max_length,
        residual_dim=config.residual_dim,
    )
    hidden_states = embeddings(input_ids)
    tied_head = LMHead(
        config.residual_dim,
        config.vocab_size,
        embeddings.token_embedding,
        tie_weights=True,
    )
    untied_head = LMHead(
        config.residual_dim,
        config.vocab_size,
        embeddings.token_embedding,
        tie_weights=False,
    )

    assert hidden_states.shape == (*input_ids.shape, config.residual_dim)
    assert tied_head(hidden_states).shape == (*input_ids.shape, config.vocab_size)
    assert untied_head(hidden_states).shape == (*input_ids.shape, config.vocab_size)


def test_hf_tokenizer_wrapper_offline():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    Tokenizer = tokenizers.Tokenizer
    WordLevel = tokenizers.models.WordLevel
    Whitespace = tokenizers.pre_tokenizers.Whitespace
    PreTrainedTokenizerFast = transformers.PreTrainedTokenizerFast

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

    assert len(model.blocks) == config.depth - 1
    assert isinstance(model.final_block, FinalResidualBlock)
    assert len(model.ema_ce_norms) == config.depth - 1
    assert len(model.ema_compressors) == config.depth - 1
    assert model.embeddings.token_embedding.num_embeddings == config.vocab_size
    assert model.embeddings.position_embedding.num_embeddings == config.max_length
    assert model.embeddings.token_embedding.embedding_dim == config.residual_dim


def test_model_uses_explicit_small_initialization():
    torch.manual_seed(0)
    model = IntertwinedHJEPA(make_config())

    assert 0.0 < model.embeddings.token_embedding.weight.std().item() < 0.1
    assert 0.0 < model.embeddings.position_embedding.weight.std().item() < 0.1
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0
        if isinstance(module, nn.RMSNorm):
            assert torch.allclose(module.weight, torch.ones_like(module.weight))
        if isinstance(module, CausalSelfAttention):
            for weight in (
                module.q_proj.weight,
                module.k_proj.weight,
                module.v_proj.weight,
                module.out_proj.weight,
            ):
                assert 0.0 < weight.std().item() < 0.1
    for block, ema_ce_norm, ema_compressor in zip(model.blocks, model.ema_ce_norms, model.ema_compressors):
        assert torch.equal(block.ce_norm.weight, ema_ce_norm.weight)
        for student_parameter, ema_parameter in zip(
            block.compressor.parameters(),
            ema_compressor.module.parameters(),
        ):
            assert torch.equal(student_parameter, ema_parameter)


def test_jepa_target_uses_ema_norm_not_live_norm():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)
    next_post_attn = outputs["post_attn_states"][1]

    with torch.no_grad():
        model.blocks[1].ce_norm.weight.fill_(2.0)

    target = model.compute_jepa_target_for_layer(0, outputs["post_attn_states"])
    ema_target = model.ema_compressors[1].module(model.ema_ce_norms[1](next_post_attn))
    live_target = model.blocks[1].compressor(model.blocks[1].ce_norm(next_post_attn))

    assert torch.allclose(target, ema_target)
    assert not torch.allclose(target, live_target)


def test_last_jepa_target_uses_frozen_output_encoder():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)
    final_post_attn = outputs["post_attn_states"][-1]

    target = model.compute_jepa_target_for_layer(len(model.blocks) - 1, outputs["post_attn_states"])
    expected = model.output_target_compressor(model.output_target_norm(final_post_attn))

    assert torch.allclose(target, expected)
    assert not target.requires_grad
    assert all(not parameter.requires_grad for parameter in model.output_target_norm.parameters())
    assert all(not parameter.requires_grad for parameter in model.output_target_compressor.parameters())


def test_load_legacy_state_without_ema_ce_norms():
    model = IntertwinedHJEPA(make_config())
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("ema_ce_norms.")
    }
    loaded = IntertwinedHJEPA(make_config())

    loaded.load_state_dict(legacy_state)

    for block, ema_ce_norm in zip(loaded.blocks, loaded.ema_ce_norms):
        assert torch.equal(block.ce_norm.weight, ema_ce_norm.weight)


def test_model_forward_returns_expected_shapes():
    config = make_config()
    model = IntertwinedHJEPA(config)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs["logits"].shape == (*input_ids.shape, config.vocab_size)
    assert outputs["final_states"].shape == (*input_ids.shape, config.residual_dim)
    assert outputs["loss"].ndim == 0
    assert outputs["loss_main"].ndim == 0
    assert outputs["loss_jepa"].ndim == 0
    assert outputs["loss_sigreg"].ndim == 0
    assert len(outputs["loss_jepa_layers"]) == config.depth - 1
    assert len(outputs["loss_sigreg_layers"]) == config.depth - 1
    assert len(outputs["z"]) == config.depth - 1
    assert len(outputs["deltas"]) == config.depth - 1
    assert len(outputs["targets"]) == config.depth - 1
    assert len(outputs["states"]) == config.depth + 1
    assert len(outputs["post_attn_states"]) == config.depth


def test_model_forward_without_labels_uses_jepa_loss():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    outputs = model(input_ids=input_ids)

    assert outputs["loss_main"] is None
    assert torch.equal(outputs["loss"], make_config().lambda_jepa * outputs["loss_jepa"])
    assert torch.equal(outputs["loss_jepa"], torch.stack(outputs["loss_jepa_layers"]).sum())
    assert torch.equal(outputs["loss_sigreg"], torch.zeros_like(outputs["loss_sigreg"]))


def test_depth_must_allow_future_layer_target():
    config = replace(make_config(), depth=1)

    try:
        IntertwinedHJEPA(config)
    except AssertionError as exc:
        assert "depth" in str(exc)
    else:
        raise AssertionError("Expected depth < 2 to be rejected")
