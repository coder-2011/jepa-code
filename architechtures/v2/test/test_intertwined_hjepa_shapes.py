from dataclasses import replace
from pathlib import Path
import warnings

import pytest
import torch
from torch import nn

from intertwined_hjepa import (
    CausalSelfAttention,
    FinalResidualBlock,
    IntertwinedBlock,
    IntertwinedConfig,
    IntertwinedHJEPA,
    RMSNorm,
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


def manual_causal_attention_output(module: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, residual_dim = x.shape
    q, k, v = module.c_attn(x).chunk(3, dim=-1)
    q = q.view(batch_size, sequence_length, module.num_heads, module.head_dim).transpose(1, 2)
    k = k.view(batch_size, sequence_length, module.num_heads, module.head_dim).transpose(1, 2)
    v = v.view(batch_size, sequence_length, module.num_heads, module.head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (module.head_dim ** -0.5)
    causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool), diagonal=1)
    probs = torch.softmax(scores.masked_fill(causal_mask, float("-inf")), dim=-1)
    attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(batch_size, sequence_length, residual_dim)
    return module.c_proj(attn_out)


class ConstantLike(nn.Module):
    def __init__(self, out_dim: int, fill_value: float):
        super().__init__()
        self.out_dim = out_dim
        self.fill_value = fill_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((*x.shape[:-1], self.out_dim), self.fill_value, dtype=x.dtype, device=x.device)


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


def test_runtime_residual_branch_scaling_applies_to_jepa_block_updates():
    block = IntertwinedBlock(
        residual_dim=4,
        compressed_dim=2,
        predictor_hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        residual_branch_scale=0.5,
    )
    block.attn_norm = nn.Identity()
    block.ce_norm = nn.Identity()
    block.attn = ConstantLike(out_dim=4, fill_value=2.0)
    block.compressor = ConstantLike(out_dim=2, fill_value=1.0)
    block.predictor = ConstantLike(out_dim=2, fill_value=0.0)
    block.projector = ConstantLike(out_dim=4, fill_value=4.0)
    x = torch.zeros(1, 3, 4)

    out = block.forward_student(x)

    assert torch.allclose(out["x_post_attn"], torch.full_like(out["x_post_attn"], 1.0))
    assert torch.allclose(out["x_next"], torch.full_like(out["x_next"], 3.0))


def test_runtime_residual_branch_scaling_applies_to_final_block_updates():
    block = FinalResidualBlock(
        residual_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        residual_branch_scale=0.5,
    )
    block.attn_norm = nn.Identity()
    block.attn = ConstantLike(out_dim=4, fill_value=2.0)
    block.mlp = ConstantLike(out_dim=4, fill_value=4.0)
    x = torch.zeros(1, 3, 4)

    out = block(x)

    assert torch.allclose(out["x_post_attn"], torch.full_like(out["x_post_attn"], 1.0))
    assert torch.allclose(out["x_next"], torch.full_like(out["x_next"], 3.0))


def test_causal_self_attention_matches_manual_reference():
    config = make_config()
    attention = CausalSelfAttention(
        residual_dim=config.residual_dim,
        num_heads=config.num_heads,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 5, config.residual_dim)

    with torch.no_grad():
        direct = attention(x)
        manual = manual_causal_attention_output(attention, x)

    assert torch.allclose(direct, manual, atol=1e-6, rtol=1e-5)


def test_causal_self_attention_ignores_future_tokens():
    config = make_config()
    attention = CausalSelfAttention(
        residual_dim=config.residual_dim,
        num_heads=config.num_heads,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 5, config.residual_dim)
    perturbed = x.clone()
    perturbed[:, -1] += 1000.0

    with torch.no_grad():
        base = attention(x)
        changed = attention(perturbed)

    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(base[:, -1], changed[:, -1])


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
    assert len(model.ema_target_encoders) == config.depth - 1
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
                module.c_attn.weight,
                module.c_proj.weight,
            ):
                assert 0.0 < weight.std().item() < 0.1
    for block, ema_encoder in zip(model.blocks, model.ema_target_encoders):
        assert not hasattr(ema_encoder, "attn")
        assert not hasattr(ema_encoder, "attn_norm")
        assert torch.equal(block.ce_norm.weight, ema_encoder.ce_norm.weight)
        for student_parameter, ema_parameter in zip(block.compressor.parameters(), ema_encoder.compressor.parameters()):
            assert torch.equal(student_parameter, ema_parameter)


def test_residual_output_projections_use_scaled_init():
    torch.manual_seed(0)
    config = make_config()
    model = IntertwinedHJEPA(config)
    target_std = 0.02 / (2.0 * config.depth) ** 0.5

    for block in model.blocks:
        assert block.attn.c_proj.weight.std().item() == pytest.approx(target_std, rel=0.35)
        assert block.projector[1].weight.std().item() == pytest.approx(target_std, rel=0.35)

    assert model.final_block.attn.c_proj.weight.std().item() == pytest.approx(target_std, rel=0.35)
    assert model.final_block.mlp[4].weight.std().item() == pytest.approx(target_std, rel=0.35)


def test_rmsnorm_mixed_bf16_input_avoids_dtype_mismatch_warning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = RMSNorm(8).to(device=device)
    x = torch.randn(2, 3, 8, device=device, dtype=torch.bfloat16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = norm(x)

    assert norm.weight.dtype == torch.float32
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()
    assert not any("Mismatch dtype between input and weight" in str(w.message) for w in caught)


def test_jepa_target_uses_same_layer_ema_ce_path_on_cached_post_attention_state():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)
    current_post_attn = outputs["post_attn_states"][0]

    with torch.no_grad():
        model.blocks[0].ce_norm.weight.fill_(2.0)
        model.blocks[1].ce_norm.weight.fill_(2.0)

    target = model.compute_raw_jepa_target_for_layer(0, outputs["post_attn_states"])
    ema_target = model.ema_target_encoders[0](current_post_attn)
    live_target = model.blocks[0].compressor(model.blocks[0].ce_norm(current_post_attn))
    next_block_target = model.ema_target_encoders[1](current_post_attn)

    assert torch.allclose(target, ema_target)
    assert torch.equal(outputs["jepa_valid_mask"][:, -1], torch.zeros_like(outputs["jepa_valid_mask"][:, -1]))
    assert not torch.allclose(target, live_target)
    assert not torch.allclose(target, next_block_target)


def test_last_jepa_target_uses_same_layer_next_token_ema_encoder():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)
    current_post_attn = outputs["post_attn_states"][len(model.blocks) - 1]

    target = model.compute_raw_jepa_target_for_layer(len(model.blocks) - 1, outputs["post_attn_states"])
    expected = model.ema_target_encoders[-1](current_post_attn)

    assert torch.allclose(target, expected)
    assert not target.requires_grad


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


def test_residual_stream_updates_decompose_per_layer():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)

    for index in range(model.config.depth):
        state_in = outputs["states"][index]
        post_attn = outputs["post_attn_states"][index]
        state_out = outputs["states"][index + 1]
        attn_update = post_attn - state_in
        block_update = state_out - post_attn
        total_update = state_out - state_in

        assert torch.allclose(total_update, attn_update + block_update)
    assert len(outputs["loss_jepa_layers"]) == model.config.depth - 1
    assert len(outputs["loss_sigreg_layers"]) == model.config.depth - 1
    assert len(outputs["z"]) == model.config.depth - 1
    assert len(outputs["deltas"]) == model.config.depth - 1
    assert len(outputs["targets"]) == model.config.depth - 1
    assert len(outputs["states"]) == model.config.depth + 1
    assert len(outputs["post_attn_states"]) == model.config.depth
    assert outputs["jepa_valid_mask"].shape == input_ids.shape
    assert outputs["jepa_valid_mask"][:, :-1].all()
    assert not outputs["jepa_valid_mask"][:, -1].any()


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
