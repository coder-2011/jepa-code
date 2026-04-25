import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from sigreg import SIGReg
from text_helpers import LMHead, TokenEmbeddings
from utils.flash_attention import flash_attn

"""
Notation used below:
    B: batch size
    L: sequence length
    D: residual width
    K: compressed JEPA width

Core v1 contract:
    h_l:              (B, L, D)
    h_l_post_attn:    (B, L, D)
    z_l:              (B, L, K)
    delta_l:          (B, L, K)
    target_z_l:       (B, L, K), same-layer EMA CE output before the next-token shift
    logits:           (B, L, vocab_size)
"""


@dataclass
class IntertwinedConfig:
    # No defaults here: architecture/loss values should come from YAML explicitly.
    vocab_size: int
    max_length: int
    residual_dim: int
    compressed_dim: int
    depth: int
    num_heads: int
    predictor_hidden_dim: int
    dropout: float
    ema_momentum: float
    lambda_jepa: float
    jepa_warmup_steps: int
    jepa_dropout_rate: float
    beta_sigreg: float
    sigreg_warmup_steps: int
    sigreg_num_slices: int
    sigreg_t_max: float
    sigreg_n_points: int
    tie_weights: bool
    rope_base: float = 10000.0
    ema_momentum_final: float = 0.996
    ema_warmup_steps: int = 0
    auxiliary_layer_start: int = 0
    auxiliary_layer_stride: int = 1
    auxiliary_target_groups: list[dict[str, object]] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IntertwinedConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        assert isinstance(values, dict), f"Expected mapping config in {path}"
        return cls(**values)


@dataclass(frozen=True)
class AuxiliaryTargetSpec:
    layer_index: int
    target_type: str
    prediction_type: str
    horizon_start: int
    horizon_end: int
    weight: float = 1.0


class SimpleCompressor(nn.Module):
    def __init__(self, residual_dim: int, compressed_dim: int, dropout: float = 0.0):
        super().__init__()
        # Small MLP that maps the residual stream from D down to JEPA width K.
        self.net = nn.Sequential(
            nn.Linear(residual_dim, compressed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(compressed_dim, compressed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RMSNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.float()
        return F.rms_norm(x.float(), self.normalized_shape, weight, self.eps).to(dtype=x.dtype)


class DeltaPredictor(nn.Module):
    def __init__(self, compressed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # Predict the same-layer next-token delta directly in compressed space.
        self.net = nn.Sequential(
            RMSNorm(compressed_dim),
            nn.Linear(compressed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, compressed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., K) -> (..., K)
        return self.net(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires an even head dimension"
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", angles.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        assert seq_len <= self.max_seq_len, f"sequence length {seq_len} exceeds RoPE cache length {self.max_seq_len}"
        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=dtype),
            self.sin_cached[:, :, :seq_len, :].to(dtype=dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0, "RoPE requires an even last dimension"
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        residual_dim: int,
        num_heads: int,
        max_length: int,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert residual_dim % num_heads == 0, "residual_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = residual_dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.dropout = dropout
        self.c_attn = nn.Linear(residual_dim, 3 * residual_dim, bias=False)
        self.c_proj = nn.Linear(residual_dim, residual_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_length, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, "x must have shape (B, L, D)"
        batch_size, sequence_length, residual_dim = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(sequence_length, dtype=q.dtype)
        q = apply_rotary_embedding(q, cos, sin)
        k = apply_rotary_embedding(k, cos, sin)
        attn_out = flash_attn.flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.c_proj(attn_out.reshape(batch_size, sequence_length, residual_dim))


def init_intertwined_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        std = float(getattr(module, "_init_std", 0.02)) if isinstance(module, nn.Linear) else 0.02
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.RMSNorm) and module.weight is not None:
        nn.init.ones_(module.weight)


@torch.no_grad()
def ema_update(ema_parameters: tuple[torch.Tensor, ...], student_parameters: tuple[torch.Tensor, ...], decay: float) -> None:
    assert len(ema_parameters) == len(student_parameters), "EMA and student parameter collections must match"
    torch._foreach_mul_(ema_parameters, decay)
    torch._foreach_add_(ema_parameters, student_parameters, alpha=1.0 - decay)

class IntertwinedBlock(nn.Module):
    def __init__(
        self,
        residual_dim: int,
        compressed_dim: int,
        predictor_hidden_dim: int,
        max_length: int,
        num_heads: int = 1,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        residual_branch_scale: float = 1.0,
    ):
        super().__init__()
        self.residual_branch_scale = float(residual_branch_scale)
        # Pre-norm causal self-attention on the residual stream.
        self.attn_norm = RMSNorm(residual_dim)
        self.attn = CausalSelfAttention(
            residual_dim=residual_dim,
            num_heads=num_heads,
            max_length=max_length,
            rope_base=rope_base,
            dropout=dropout,
        )
        # Compressor path: D -> K, predictor path: K -> K.
        self.ce_norm = RMSNorm(residual_dim)
        self.compressor = SimpleCompressor(residual_dim, compressed_dim, dropout=dropout)
        self.predictor = DeltaPredictor(compressed_dim, predictor_hidden_dim, dropout=dropout)
        self.transition_mlp = nn.Sequential(
            RMSNorm(residual_dim),
            nn.Linear(residual_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden_dim, residual_dim),
        )

    def encode_context(self, x_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x_l.ndim == 3, "x_l must have shape (B, L, D)"
        x_l_post_attn = x_l + self.residual_branch_scale * self.attn(self.attn_norm(x_l))
        z_l = self.compressor(self.ce_norm(x_l_post_attn))
        return x_l_post_attn, z_l

    def forward_student(self, x_l: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_l: (B, L, D)

        Returns:
            x_next:      same leading shape as x_l, residual width D
            x_post_attn: same leading shape as x_l, residual width D
            z:           same leading shape as x_l, compressed width K
            delta:       same leading shape as x_l, compressed width K
        """
        assert x_l.ndim == 3, "x_l must have shape (B, L, D)"
        x_l_post_attn, z_l = self.encode_context(x_l)
        delta_l = self.predictor(z_l)
        # Keep the JEPA latent auxiliary-only; the residual stream has its own transition path.
        update_l = self.transition_mlp(x_l_post_attn)
        x_next = x_l_post_attn + self.residual_branch_scale * update_l

        return {
            "x_next": x_next,
            "x_post_attn": x_l_post_attn,
            "z": z_l,
            "delta": delta_l,
        }


class JEPAEncoder(nn.Module):
    def __init__(
        self,
        residual_dim: int,
        compressed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ce_norm = RMSNorm(residual_dim)
        self.compressor = SimpleCompressor(residual_dim, compressed_dim, dropout=dropout)

    def forward(self, x_l_post_attn: torch.Tensor) -> torch.Tensor:
        assert x_l_post_attn.ndim == 3, "x_l_post_attn must have shape (B, L, D)"
        return self.compressor(self.ce_norm(x_l_post_attn))


class FinalResidualBlock(nn.Module):
    def __init__(
        self,
        residual_dim: int,
        hidden_dim: int,
        max_length: int,
        num_heads: int = 1,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        residual_branch_scale: float = 1.0,
    ):
        super().__init__()
        self.residual_branch_scale = float(residual_branch_scale)
        self.attn_norm = RMSNorm(residual_dim)
        self.attn = CausalSelfAttention(
            residual_dim=residual_dim,
            num_heads=num_heads,
            max_length=max_length,
            rope_base=rope_base,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            RMSNorm(residual_dim),
            nn.Linear(residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, residual_dim),
        )

    def forward(self, x_l: torch.Tensor) -> dict[str, torch.Tensor]:
        x_l_normed = self.attn_norm(x_l)
        attn_out = self.residual_branch_scale * self.attn(x_l_normed)
        x_post_attn = x_l + attn_out
        return {
            "x_next": x_post_attn + self.residual_branch_scale * self.mlp(x_post_attn),
            "x_post_attn": x_post_attn,
        }


def jepa_delta_loss(
    delta_l: torch.Tensor,
    z_l: torch.Tensor,
    target_z_l: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Loss for already-aligned JEPA tensors.

    In the model forward path, the caller aligns same-layer targets by passing
    `z_l[:, :-1]`, `delta_l[:, :-1]`, and `target_z_l[:, 1:]`.
    """
    assert delta_l.shape == z_l.shape == target_z_l.shape, "delta_l, z_l, and target_z_l must have exactly the same shape"
    assert delta_l.ndim >= 2, "delta_l, z_l, and target_z_l must have at least one sample axis and one feature axis"

    normalized_z_l = rms_normalize_last_dim(z_l)
    normalized_delta_l = rms_normalize_last_dim(delta_l)
    normalized_target_delta_l = rms_normalize_last_dim(target_z_l.detach()) - normalized_z_l

    if valid_mask is None:
        return F.mse_loss(normalized_delta_l, normalized_target_delta_l)

    assert valid_mask.shape == delta_l.shape[:-1], "valid_mask must match the leading shape of delta_l"
    assert valid_mask.any(), "valid_mask selects no JEPA loss positions"

    error = F.mse_loss(normalized_delta_l, normalized_target_delta_l, reduction="none")
    return error[valid_mask].mean()


def jepa_state_loss(
    z_l: torch.Tensor,
    target_z_l: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Directly align a student state with an already-aligned future target."""
    assert z_l.shape == target_z_l.shape, "z_l and target_z_l must have exactly the same shape"
    assert z_l.ndim >= 2, "z_l and target_z_l must have at least one sample axis and one feature axis"

    normalized_z_l = rms_normalize_last_dim(z_l)
    normalized_target_z_l = rms_normalize_last_dim(target_z_l.detach())

    if valid_mask is None:
        return F.mse_loss(normalized_z_l, normalized_target_z_l)

    assert valid_mask.shape == z_l.shape[:-1], "valid_mask must match the leading shape of z_l"
    assert valid_mask.any(), "valid_mask selects no JEPA loss positions"

    error = F.mse_loss(normalized_z_l, normalized_target_z_l, reduction="none")
    return error[valid_mask].mean()


def next_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # logits: (B, L, vocab_size), labels/valid_mask: (B, L)
    # The dataloader already shifts targets so labels[:, t] is the next token for logits[:, t].
    if valid_mask is not None:
        labels = labels.masked_fill(~valid_mask.to(torch.bool), -100)

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def next_token_jepa_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Mask positions that have a next-token JEPA target."""
    assert input_ids.ndim == 2, "input_ids must have shape (B, L)"
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    mask[:, -1] = False
    return mask


def warmup_weight(weight: float, step: int | None, warmup_steps: int) -> float:
    # Scalar schedule only; it does not touch any tensor shapes.
    if warmup_steps <= 0 or step is None:
        return weight
    return weight * min(1.0, (step + 1) / warmup_steps)


def rms_normalize_last_dim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * scale.to(device=x.device, dtype=x.dtype)


def split_scalars(x: torch.Tensor) -> list[torch.Tensor]:
    assert x.ndim == 1, "split_scalars expects a 1D tensor"
    # `unbind()` returns scalar views, which can trip torch.compile alias handling.
    return [x[index].clone() for index in range(x.shape[0])]


def auxiliary_layer_indices(num_layers: int, start: int, stride: int) -> tuple[int, ...]:
    assert num_layers >= 0, "num_layers must be non-negative"
    assert 0 <= start <= num_layers, "auxiliary_layer_start must select a valid JEPA block boundary"
    assert stride > 0, "auxiliary_layer_stride must be positive"
    return tuple(range(start, num_layers, stride))


def auxiliary_target_specs(
    num_layers: int,
    groups: list[dict[str, object]] | None,
    default_layer_indices: tuple[int, ...],
) -> tuple[AuxiliaryTargetSpec, ...]:
    if groups is None:
        return tuple(
            AuxiliaryTargetSpec(
                layer_index=layer_index,
                target_type="same_layer",
                prediction_type="delta",
                horizon_start=1,
                horizon_end=1,
            )
            for layer_index in default_layer_indices
        )

    specs = []
    seen_layers = set()
    valid_target_types = {"same_layer", "final_layer", "none", "off"}
    valid_prediction_types = {"delta", "state", "direct"}
    for group in groups:
        assert isinstance(group, dict), "each auxiliary target group must be a mapping"
        target_type = str(group.get("target_type", group.get("target", "same_layer")))
        assert target_type in valid_target_types, f"unsupported auxiliary target_type {target_type!r}"
        prediction_type = str(group.get("prediction_type", group.get("prediction", "delta")))
        assert prediction_type in valid_prediction_types, f"unsupported auxiliary prediction_type {prediction_type!r}"
        if prediction_type == "direct":
            prediction_type = "state"
        weight = float(group.get("weight", 1.0))
        assert weight >= 0.0, "auxiliary target weight must be non-negative"
        layers = group.get("layers")
        horizon = group.get("horizon", [1, 1])
        assert isinstance(layers, list) and layers, "auxiliary target group must define non-empty layers"
        assert isinstance(horizon, list) and len(horizon) == 2, "auxiliary target horizon must be [start, end]"
        horizon_start = int(horizon[0])
        horizon_end = int(horizon[1])
        assert 1 <= horizon_start <= horizon_end, "auxiliary target horizon must satisfy 1 <= start <= end"

        if target_type in {"none", "off"} or weight == 0.0:
            continue
        for raw_layer_index in layers:
            layer_index = int(raw_layer_index)
            assert 0 <= layer_index < num_layers, "auxiliary target layer index is out of range"
            assert layer_index not in seen_layers, "auxiliary target groups must not overlap layers"
            seen_layers.add(layer_index)
            specs.append(
                AuxiliaryTargetSpec(
                    layer_index=layer_index,
                    target_type=target_type,
                    prediction_type=prediction_type,
                    horizon_start=horizon_start,
                    horizon_end=horizon_end,
                    weight=weight,
                )
            )
    return tuple(sorted(specs, key=lambda spec: spec.layer_index))


def future_target_summary(
    target_sequence: torch.Tensor,
    current_valid_mask: torch.Tensor,
    future_valid_mask: torch.Tensor,
    horizon_start: int,
    horizon_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert target_sequence.ndim == 3, "target_sequence must have shape (B, L, K)"
    assert current_valid_mask.shape == target_sequence.shape[:2], "current_valid_mask must have shape (B, L)"
    assert future_valid_mask.shape == target_sequence.shape[:2], "future_valid_mask must have shape (B, L)"
    total = target_sequence.new_zeros(target_sequence.shape)
    counts = target_sequence.new_zeros(target_sequence.shape[:2])
    for horizon in range(horizon_start, horizon_end + 1):
        if horizon >= target_sequence.shape[1]:
            continue
        valid = current_valid_mask[:, :-horizon] & future_valid_mask[:, horizon:]
        total[:, :-horizon] = total[:, :-horizon] + target_sequence[:, horizon:] * valid.unsqueeze(-1)
        counts[:, :-horizon] = counts[:, :-horizon] + valid.to(dtype=counts.dtype)
    summary = total / counts.clamp_min(1).unsqueeze(-1)
    return summary, counts > 0


def load_intertwined_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    migrated_state_dict = dict(state_dict)
    ignored_legacy_keys = []
    legacy_position_key = "embeddings.position_embedding.weight"
    if legacy_position_key in migrated_state_dict:
        migrated_state_dict.pop(legacy_position_key)
        ignored_legacy_keys.append(legacy_position_key)
    incompatible = model.load_state_dict(migrated_state_dict, strict=False)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, (
        "Unexpected checkpoint mismatch after RoPE migration: "
        f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
    )
    return ignored_legacy_keys


class IntertwinedHJEPA(nn.Module):
    """
    For depth=N, the model has N-1 JEPA blocks and one normal final residual block.

    h_l_post_attn = h_l + Attention_l(RMSNorm(h_l))
    z_l = CE_l(RMSNorm(h_l_post_attn))
    d_l = Pred_l(z_l)
    h_{l+1} = h_l_post_attn + MLP_l(h_l_post_attn)

    target_z_l = sg(EMA_CE_l(h_l_post_attn)) for the same JEPA block
    L_jepa_l = MSE(d_l[:, t], target_z_l[:, t+1] - z_l[:, t])
    """

    def __init__(self, config: IntertwinedConfig):
        super().__init__()
        assert config.depth >= 2, "depth must be at least 2"
        assert 0.0 <= config.jepa_dropout_rate <= 1.0, "jepa_dropout_rate must be between 0 and 1"
        assert 0.0 <= config.ema_momentum <= 1.0 and 0.0 <= config.ema_momentum_final <= 1.0 and config.ema_warmup_steps >= 0, (
            "EMA momentum must be in [0, 1] and ema_warmup_steps must be non-negative"
        )

        self.config = config
        self.ema_momentum = float(config.ema_momentum)
        self.ema_momentum_final = float(config.ema_momentum_final)
        self.ema_warmup_steps = int(config.ema_warmup_steps)
        residual_branch_scale = 1.0 / math.sqrt(config.depth)

        # Token embeddings stay learned; positions enter through RoPE inside attention.
        self.embeddings = TokenEmbeddings(
            vocab_size=config.vocab_size,
            max_length=config.max_length,
            residual_dim=config.residual_dim,
        )
        jepa_depth = config.depth - 1
        self.blocks = nn.ModuleList(
            [
                IntertwinedBlock(
                    residual_dim=config.residual_dim,
                    compressed_dim=config.compressed_dim,
                    predictor_hidden_dim=config.predictor_hidden_dim,
                    max_length=config.max_length,
                    num_heads=config.num_heads,
                    rope_base=config.rope_base,
                    dropout=config.dropout,
                    residual_branch_scale=residual_branch_scale,
                )
                for _ in range(jepa_depth)
            ]
        )
        self.auxiliary_layer_start = int(config.auxiliary_layer_start)
        self.auxiliary_layer_stride = int(config.auxiliary_layer_stride)
        default_auxiliary_layer_indices = auxiliary_layer_indices(
            len(self.blocks),
            self.auxiliary_layer_start,
            self.auxiliary_layer_stride,
        )
        self.auxiliary_target_specs = auxiliary_target_specs(
            len(self.blocks),
            config.auxiliary_target_groups,
            default_auxiliary_layer_indices,
        )
        self.auxiliary_layer_indices = tuple(spec.layer_index for spec in self.auxiliary_target_specs)
        self.final_block = FinalResidualBlock(
            residual_dim=config.residual_dim,
            hidden_dim=config.predictor_hidden_dim,
            max_length=config.max_length,
            num_heads=config.num_heads,
            rope_base=config.rope_base,
            dropout=config.dropout,
            residual_branch_scale=residual_branch_scale,
        )
        self.final_norm = RMSNorm(config.residual_dim)
        self.sigreg = SIGReg(
            num_slices=config.sigreg_num_slices,
            t_max=config.sigreg_t_max,
            knots=config.sigreg_n_points,
        )
        self.lm_head = LMHead(
            residual_dim=config.residual_dim,
            vocab_size=config.vocab_size,
            token_embedding=self.embeddings.token_embedding,
            tie_weights=config.tie_weights,
        )
        residual_output_std = 0.02 / math.sqrt(2.0 * config.depth)
        for block in self.blocks:
            block.attn.c_proj._init_std = residual_output_std
            block.transition_mlp[4]._init_std = residual_output_std
        self.final_block.attn.c_proj._init_std = residual_output_std
        self.final_block.mlp[4]._init_std = residual_output_std
        self.apply(init_intertwined_weights)
        self.ema_target_encoders = nn.ModuleList(
            [
                JEPAEncoder(
                    residual_dim=config.residual_dim,
                    compressed_dim=config.compressed_dim,
                    dropout=config.dropout,
                )
                for _ in range(jepa_depth)
            ]
        )
        for student_block, ema_encoder in zip(self.blocks, self.ema_target_encoders):
            ema_encoder.load_state_dict(
                {
                    "ce_norm.weight": student_block.ce_norm.weight.detach().clone(),
                    "compressor.net.0.weight": student_block.compressor.net[0].weight.detach().clone(),
                    "compressor.net.0.bias": student_block.compressor.net[0].bias.detach().clone(),
                    "compressor.net.3.weight": student_block.compressor.net[3].weight.detach().clone(),
                    "compressor.net.3.bias": student_block.compressor.net[3].bias.detach().clone(),
                }
            )
            ema_encoder.requires_grad_(False)
            ema_encoder.eval()

    def student_parameters(self):
        # EMA parameters are frozen, so this naturally returns only trainable student weights.
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def ema_momentum_at_step(self, step: int | None) -> float:
        if step is None or self.ema_warmup_steps <= 0 or self.ema_momentum == self.ema_momentum_final:
            return self.ema_momentum
        progress = min(1.0, (step + 1) / self.ema_warmup_steps)
        return self.ema_momentum + (self.ema_momentum_final - self.ema_momentum) * progress

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L) -> h_0: (B, L, D)
        return self.embeddings(input_ids)

    @torch.no_grad()
    def compute_raw_jepa_target_for_layer(
        self,
        layer_index: int,
        post_attn_states: list[torch.Tensor],
        final_states: torch.Tensor | None = None,
        target_type: str = "same_layer",
    ) -> torch.Tensor:
        """Return the unshifted EMA CE target sequence for a JEPA block."""
        assert 0 <= layer_index < len(self.blocks), "layer_index must point to a JEPA block"
        if target_type == "same_layer":
            source_states = post_attn_states[layer_index]
        elif target_type == "final_layer":
            assert final_states is not None, "final_layer auxiliary targets require final_states"
            source_states = final_states
        else:
            raise AssertionError(f"unsupported auxiliary target_type {target_type!r}")
        target_z_l = self.ema_target_encoders[layer_index](source_states)
        return target_z_l.detach()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        step: int | None = None,
        compute_aux_losses: bool = True,
        lambda_jepa_scale: torch.Tensor | float | None = None,
        beta_sigreg_scale: torch.Tensor | float | None = None,
    ) -> dict[str, object]:
        """
        Args:
            input_ids:  (B, L)
            labels:     (B, L), optional next-token labels
            valid_mask: (B, L), optional loss mask on token positions
            compute_aux_losses: when false, skip teacher-target JEPA and SIGReg loss computation

        Returns:
            logits:            (B, L, vocab_size)
            final_states:      (B, L, D)
            states:            depth + 1 residual states, each (B, L, D)
            post_attn_states:  depth states, each (B, L, D)
            z:                 depth - 1 JEPA compressed states, each (B, L, K)
            deltas:            depth - 1 predicted deltas, each (B, L, K)
            targets:           depth - 1 raw stopped same-layer EMA targets, each (B, L, K)
            loss_sigreg_layers: depth - 1 local SIGReg losses
        """
        # h starts as h_0: dense token states of shape (B, L, D).
        h = self.embed(input_ids)
        states = []
        post_attn_states = []
        compressed = []
        deltas = []

        # JEPA blocks preserve the dense residual stream and expose compressed local predictions.
        for block in self.blocks:
            states.append(h)
            out = block.forward_student(h)
            h = out["x_next"]
            post_attn_states.append(out["x_post_attn"])
            compressed.append(out["z"])
            deltas.append(out["delta"])

        states.append(h)
        final_out = self.final_block(h)
        h = final_out["x_next"]
        post_attn_states.append(final_out["x_post_attn"])
        states.append(h)
        # Final norm stays in the model; the helper only does the D -> vocab projection.
        logits = self.lm_head(self.final_norm(h))
        sequence_valid_mask = None if valid_mask is None else valid_mask.to(torch.bool)
        jepa_valid_mask = next_token_jepa_mask(input_ids)
        jepa_current_valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        jepa_future_valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if sequence_valid_mask is not None:
            jepa_valid_mask &= sequence_valid_mask
            jepa_current_valid_mask &= sequence_valid_mask

        if compute_aux_losses:
            zero_loss = logits.new_zeros(())
            jepa_losses = [zero_loss.clone() for _ in range(len(self.blocks))]
            sigreg_losses = [zero_loss.clone() for _ in range(len(self.blocks))]
            targets = [z.detach() for z in compressed]
            active_layer_indices = self.auxiliary_layer_indices
            jepa_position_count = logits.new_zeros(())

            if active_layer_indices:
                z_stack = torch.stack([compressed[index] for index in active_layer_indices])
                for spec in self.auxiliary_target_specs:
                    layer_index = spec.layer_index
                    target_sequence = self.compute_raw_jepa_target_for_layer(
                        layer_index,
                        post_attn_states,
                        final_states=h,
                        target_type=spec.target_type,
                    )
                    targets[layer_index] = target_sequence
                    target_summary, layer_mask = future_target_summary(
                        target_sequence,
                        jepa_current_valid_mask,
                        jepa_future_valid_mask,
                        spec.horizon_start,
                        spec.horizon_end,
                    )
                    assert layer_mask.any(), (
                        "auxiliary target horizon selected no valid positions: "
                        f"layer={layer_index}, horizon=[{spec.horizon_start}, {spec.horizon_end}]"
                    )
                    normalized_z = rms_normalize_last_dim(compressed[layer_index])
                    normalized_target = rms_normalize_last_dim(target_summary.detach())
                    if spec.prediction_type == "delta":
                        normalized_delta = rms_normalize_last_dim(deltas[layer_index])
                        prediction_error = normalized_delta - (normalized_target - normalized_z)
                    elif spec.prediction_type == "state":
                        prediction_error = normalized_z - normalized_target
                    else:
                        raise AssertionError(f"unsupported auxiliary prediction_type {spec.prediction_type!r}")
                    jepa_error = prediction_error.square()
                    mask = layer_mask.unsqueeze(-1)
                    layer_loss = (jepa_error * mask).sum() / mask.sum().mul(jepa_error.shape[-1])
                    jepa_losses[layer_index] = layer_loss * spec.weight
                    jepa_position_count = jepa_position_count + layer_mask.sum()

                if self.config.beta_sigreg > 0:
                    sigreg_input = rms_normalize_last_dim(z_stack)
                    if sequence_valid_mask is None:
                        sigreg_loss_by_active_layer = self.sigreg(sigreg_input, per_layer=True)
                    else:
                        assert sequence_valid_mask.shape == sigreg_input.shape[1:-1], (
                            "valid_mask must match batch and sequence dimensions of SIGReg inputs"
                        )
                        sigreg_mask = sequence_valid_mask.unsqueeze(0).expand(len(active_layer_indices), -1, -1)
                        sigreg_loss_by_active_layer = self.sigreg(sigreg_input, sample_mask=sigreg_mask, per_layer=True)
                    for loss, spec in zip(split_scalars(sigreg_loss_by_active_layer), self.auxiliary_target_specs):
                        sigreg_losses[spec.layer_index] = loss * spec.weight
        else:
            zero_loss = logits.new_zeros(())
            # Keep output structure stable on dropout steps without running the teacher path.
            targets = [z.detach() for z in compressed]
            jepa_losses = [zero_loss.clone() for _ in range(len(self.blocks))]
            sigreg_losses = [zero_loss.clone() for _ in range(len(self.blocks))]
            jepa_position_count = zero_loss.clone()

        # Keep per-layer JEPA losses explicit and sum them into the total objective.
        loss_jepa = torch.stack(jepa_losses).sum()
        if not sigreg_losses:
            sigreg_losses = [logits.new_zeros(()) for _ in range(len(self.blocks))]
        loss_sigreg = torch.stack(sigreg_losses).sum()
        lambda_eff = (
            warmup_weight(self.config.lambda_jepa, step, self.config.jepa_warmup_steps)
            if lambda_jepa_scale is None
            else lambda_jepa_scale
        )
        beta_eff = (
            warmup_weight(self.config.beta_sigreg, step, self.config.sigreg_warmup_steps)
            if beta_sigreg_scale is None
            else beta_sigreg_scale
        )
        if not compute_aux_losses:
            lambda_eff = 0.0
            beta_eff = 0.0
        if not torch.is_tensor(lambda_eff):
            lambda_eff = logits.new_tensor(lambda_eff)
        else:
            lambda_eff = lambda_eff.to(device=logits.device, dtype=logits.dtype)
        if not torch.is_tensor(beta_eff):
            beta_eff = logits.new_tensor(beta_eff)
        else:
            beta_eff = beta_eff.to(device=logits.device, dtype=logits.dtype)
        loss_main = next_token_loss(logits, labels, valid_mask) if labels is not None else None
        loss = (loss_main if loss_main is not None else logits.new_zeros(())) + lambda_eff * loss_jepa + beta_eff * loss_sigreg

        # Keep a few cheap summaries around for smoke tests and debugging.
        z_stds = [z.detach().float().reshape(-1, z.shape[-1]).std(dim=0, unbiased=False) for z in compressed]
        diagnostics = {
            "compute_aux_losses": compute_aux_losses,
            "auxiliary_layer_start": torch.tensor(self.auxiliary_layer_start, device=logits.device),
            "auxiliary_layer_stride": torch.tensor(self.auxiliary_layer_stride, device=logits.device),
            "num_auxiliary_layers": torch.tensor(len(self.auxiliary_layer_indices), device=logits.device),
            "auxiliary_target_horizon_start": [
                torch.tensor(spec.horizon_start, device=logits.device) for spec in self.auxiliary_target_specs
            ],
            "auxiliary_target_horizon_end": [
                torch.tensor(spec.horizon_end, device=logits.device) for spec in self.auxiliary_target_specs
            ],
            "auxiliary_target_weight": [
                torch.tensor(spec.weight, device=logits.device) for spec in self.auxiliary_target_specs
            ],
            "auxiliary_prediction_type": [spec.prediction_type for spec in self.auxiliary_target_specs],
            "lambda_jepa": lambda_eff.detach(),
            "beta_sigreg": beta_eff.detach(),
            "loss_jepa_layers": [loss.detach() for loss in jepa_losses],
            "loss_sigreg_layers": [loss.detach() for loss in sigreg_losses],
            "jepa_valid_fraction": jepa_valid_mask.float().mean().detach(),
            "jepa_positions": jepa_valid_mask.sum().detach(),
            "auxiliary_target_positions": jepa_position_count.detach(),
            "z_variance": [z.detach().float().var() for z in compressed],
            "z_std_mean": [std.mean() for std in z_stds],
            "z_std_min": [std.min() for std in z_stds],
            "delta_norm": [delta.detach().float().norm() for delta in deltas],
        }

        return {
            "loss": loss,
            "loss_main": loss_main,
            "loss_jepa": loss_jepa,
            "loss_jepa_layers": jepa_losses,
            "loss_sigreg": loss_sigreg,
            "loss_sigreg_layers": sigreg_losses,
            "logits": logits,
            "final_states": h,
            "states": states,
            "post_attn_states": post_attn_states,
            "z": compressed,
            "deltas": deltas,
            "targets": targets,
            "jepa_valid_mask": jepa_valid_mask,
            "diagnostics": diagnostics,
        }

    @torch.no_grad()
    def update_ema(self, step: int | None = None) -> None:
        # Call after optimizer.step() so the EMA teacher lags the student encoder path.
        momentum = self.ema_momentum_at_step(step)
        ema_parameters = []
        student_parameters = []
        for ema_encoder, block in zip(self.ema_target_encoders, self.blocks):
            ema_parameters.extend((ema_encoder.ce_norm.weight, *ema_encoder.compressor.parameters()))
            student_parameters.extend((block.ce_norm.weight, *block.compressor.parameters()))
        ema_update(tuple(ema_parameters), tuple(student_parameters), momentum)
