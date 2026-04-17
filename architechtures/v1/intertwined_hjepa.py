from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim.swa_utils import AveragedModel

from sigreg import SlicedEppsPulleySIGReg
from text_helpers import LMHead, TokenEmbeddings

_flash_attn_func = None
for _module_name in ("flash_attn.cute", "flash_attn"):
    try:
        _flash_attn_func = __import__(_module_name, fromlist=["flash_attn_func"]).flash_attn_func
        break
    except (ImportError, AttributeError):
        pass

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
    target_z_l:       (B, L, K)
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
    ema_momentum_final: float = 0.996
    ema_warmup_steps: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IntertwinedConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        assert isinstance(values, dict), f"Expected mapping config in {path}"
        return cls(**values)


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
        # Predict the "one-layer-future" delta directly in compressed space.
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


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        residual_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert residual_dim % num_heads == 0, "residual_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = residual_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(residual_dim, residual_dim, bias=False)
        self.k_proj = nn.Linear(residual_dim, residual_dim, bias=False)
        self.v_proj = nn.Linear(residual_dim, residual_dim, bias=False)
        self.out_proj = nn.Linear(residual_dim, residual_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_sequence = x.ndim == 2
        if squeeze_sequence:
            x = x.unsqueeze(1)

        batch_size, sequence_length, residual_dim = x.shape
        q = self.q_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)

        if _flash_attn_func is not None and q.is_cuda and q.dtype in {torch.float16, torch.bfloat16}:
            attn_out = _flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            ).transpose(1, 2)

        attn_out = attn_out.reshape(batch_size, sequence_length, residual_dim)
        attn_out = self.out_proj(attn_out)

        if squeeze_sequence:
            attn_out = attn_out.squeeze(1)
        return attn_out


def init_intertwined_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.RMSNorm) and module.weight is not None:
        nn.init.ones_(module.weight)


class IntertwinedBlock(nn.Module):
    def __init__(
        self,
        residual_dim: int,
        compressed_dim: int,
        predictor_hidden_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Pre-norm causal self-attention on the residual stream.
        self.attn_norm = RMSNorm(residual_dim)
        self.attn = CausalSelfAttention(residual_dim=residual_dim, num_heads=num_heads, dropout=dropout)
        # Compressor path: D -> K, predictor path: K -> K, projector path: K -> D.
        self.ce_norm = RMSNorm(residual_dim)
        self.compressor = SimpleCompressor(residual_dim, compressed_dim, dropout=dropout)
        self.predictor = DeltaPredictor(compressed_dim, predictor_hidden_dim, dropout=dropout)
        self.projector = nn.Sequential(
            RMSNorm(compressed_dim),
            nn.Linear(compressed_dim, residual_dim),
        )

    def forward_student(self, x_l: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_l: (B, L, D) during training, or (B, D) for a single-token step.

        Returns:
            x_next:      same leading shape as x_l, residual width D
            x_post_attn: same leading shape as x_l, residual width D
            z:           same leading shape as x_l, compressed width K
            delta:       same leading shape as x_l, compressed width K
        """
        # Keep a tiny decode path for single-token stepping without a separate codepath.
        squeeze_sequence = x_l.ndim == 2
        if squeeze_sequence:
            x_l = x_l.unsqueeze(1)

        x_l_normed = self.attn_norm(x_l)
        attn_out = self.attn(x_l_normed)
        # h_l_post_attn is both the residual update output and the input to the compressor.
        x_l_post_attn = x_l + attn_out
        z_l = self.compressor(self.ce_norm(x_l_post_attn))
        delta_l = self.predictor(z_l)
        # Inject the compressed prediction back into the D-wide residual stream.
        update_l = self.projector(z_l + delta_l)
        x_next = x_l_post_attn + update_l

        # Restore the caller's rank if we entered through the single-token path.
        if squeeze_sequence:
            x_next = x_next.squeeze(1)
            x_l_post_attn = x_l_post_attn.squeeze(1)
            z_l = z_l.squeeze(1)
            delta_l = delta_l.squeeze(1)

        return {
            "x_next": x_next,
            "x_post_attn": x_l_post_attn,
            "z": z_l,
            "delta": delta_l,
        }


class FinalResidualBlock(nn.Module):
    def __init__(
        self,
        residual_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(residual_dim)
        self.attn = CausalSelfAttention(residual_dim=residual_dim, num_heads=num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            RMSNorm(residual_dim),
            nn.Linear(residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, residual_dim),
        )

    def forward(self, x_l: torch.Tensor) -> dict[str, torch.Tensor]:
        x_l_normed = self.attn_norm(x_l)
        attn_out = self.attn(x_l_normed)
        x_post_attn = x_l + attn_out
        return {
            "x_next": x_post_attn + self.mlp(x_post_attn),
            "x_post_attn": x_post_attn,
        }


def jepa_delta_loss(
    delta_l: torch.Tensor,
    z_l: torch.Tensor,
    target_z_l: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Keep the JEPA contract exact: no silent broadcasting across batch, sequence, or compressed width.
    if delta_l.shape != z_l.shape or delta_l.shape != target_z_l.shape:
        raise ValueError("delta_l, z_l, and target_z_l must have exactly the same shape")

    # The EMA target is stopped, while z_l stays live so JEPA trains the CE path.
    if valid_mask is None:
        return F.mse_loss(delta_l, target_z_l.detach() - z_l)

    assert valid_mask.shape == delta_l.shape[:-1], "valid_mask must match the leading shape of delta_l"
    if not valid_mask.any():
        raise ValueError("valid_mask selects no JEPA loss positions")

    # valid_mask is over token positions (B, L); broadcast it across the compressed width K.
    error = F.mse_loss(delta_l, target_z_l.detach() - z_l, reduction="none")
    expanded_mask = valid_mask.unsqueeze(-1).expand_as(error)
    return error.masked_select(expanded_mask).mean()


def next_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # logits: (B, L, vocab_size), labels/valid_mask: (B, L)
    # We predict token t+1 from position t, so the loss sees B * (L - 1) rows after shifting.
    assert logits.shape[1] >= 2, "next-token loss requires sequence length >= 2"
    if valid_mask is not None:
        # The mask applies to the predicted token positions, so it shifts with the labels.
        labels = labels[:, 1:].masked_fill(~valid_mask[:, 1:].to(torch.bool), -100)
    else:
        labels = labels[:, 1:]

    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def warmup_weight(weight: float, step: int | None, warmup_steps: int) -> float:
    # Scalar schedule only; it does not touch any tensor shapes.
    if warmup_steps <= 0 or step is None:
        return weight
    return weight * min(1.0, (step + 1) / warmup_steps)


class IntertwinedHJEPA(nn.Module):
    """
    For depth=N, the model has N-1 JEPA blocks and one normal final residual block.

    h_l_post_attn = h_l + Attention_l(RMSNorm(h_l))
    z_l = CE_l(RMSNorm(h_l_post_attn))
    d_l = Pred_l(z_l)
    h_{l+1} = h_l_post_attn + Proj_l(z_l + d_l)

    target_z_l = sg(CEbar_{l+1}(h_{l+1}_post_attn)) or sg(T_out(h_final_post_attn))
    L_jepa_l = MSE(d_l, target_z_l - z_l)
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

        # Plain learned token + position embeddings for v1.
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
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(jepa_depth)
            ]
        )
        self.final_block = FinalResidualBlock(
            residual_dim=config.residual_dim,
            hidden_dim=config.predictor_hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.final_norm = RMSNorm(config.residual_dim)
        self.sigreg = SlicedEppsPulleySIGReg(
            num_slices=config.sigreg_num_slices,
            t_max=config.sigreg_t_max,
            n_points=config.sigreg_n_points,
        )
        self.lm_head = LMHead(
            residual_dim=config.residual_dim,
            vocab_size=config.vocab_size,
            token_embedding=self.embeddings.token_embedding,
            tie_weights=config.tie_weights,
        )
        self.output_target_norm = RMSNorm(config.residual_dim)
        self.output_target_compressor = SimpleCompressor(
            config.residual_dim,
            config.compressed_dim,
            dropout=0.0,
        )
        self.apply(init_intertwined_weights)
        self.output_target_norm.requires_grad_(False)
        self.output_target_norm.eval()
        self.output_target_compressor.requires_grad_(False)
        self.output_target_compressor.eval()
        # Track EMA copies of the full CE path: norm + compressor.
        self.ema_ce_norms = nn.ModuleList(RMSNorm(config.residual_dim) for _ in self.blocks)
        self.ema_compressors = nn.ModuleList(AveragedModel(block.compressor, use_buffers=False) for block in self.blocks)
        for ema_ce_norm, ema_compressor, block in zip(self.ema_ce_norms, self.ema_compressors, self.blocks):
            ema_ce_norm.load_state_dict(block.ce_norm.state_dict())
            ema_ce_norm.requires_grad_(False)
            ema_ce_norm.eval()
            ema_compressor.requires_grad_(False)
            ema_compressor.eval()

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
    def compute_jepa_target_for_layer(
        self,
        layer_index: int,
        post_attn_states: list[torch.Tensor],
    ) -> torch.Tensor:
        assert 0 <= layer_index < len(self.blocks), "layer_index must point to a JEPA block"
        next_post_attn = post_attn_states[layer_index + 1]
        if layer_index + 1 < len(self.blocks):
            ema_norm = self.ema_ce_norms[layer_index + 1]
            target_z_l = self.ema_compressors[layer_index + 1].module(ema_norm(next_post_attn))
        else:
            target_z_l = self.output_target_compressor(self.output_target_norm(next_post_attn))
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
            targets:           depth - 1 stopped targets, each (B, L, K)
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

        if compute_aux_losses:
            targets = []
            jepa_losses = []
            sigreg_losses = []
            for layer_index in range(len(self.blocks)):
                target_z_l = self.compute_jepa_target_for_layer(layer_index, post_attn_states)
                targets.append(target_z_l)
                jepa_losses.append(
                    jepa_delta_loss(
                        deltas[layer_index],
                        compressed[layer_index],
                        target_z_l,
                        valid_mask=valid_mask,
                    )
                )
            if self.config.beta_sigreg > 0:
                for layer_index in range(len(self.blocks)):
                    sigreg_input_l = compressed[layer_index]
                    if valid_mask is not None:
                        assert valid_mask.shape == sigreg_input_l.shape[:-1], (
                            "valid_mask must match the leading shape of SIGReg inputs"
                        )
                        sigreg_input_l = sigreg_input_l[valid_mask.to(torch.bool)]
                    sigreg_losses.append(self.sigreg(sigreg_input_l))
        else:
            zero_loss = logits.new_zeros(())
            # Keep output structure stable on dropout steps without running the teacher path.
            targets = [z.detach() for z in compressed]
            jepa_losses = [zero_loss.clone() for _ in range(len(self.blocks))]
            sigreg_losses = [zero_loss.clone() for _ in range(len(self.blocks))]

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
            "lambda_jepa": lambda_eff.detach(),
            "beta_sigreg": beta_eff.detach(),
            "loss_jepa_layers": [loss.detach() for loss in jepa_losses],
            "loss_sigreg_layers": [loss.detach() for loss in sigreg_losses],
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
            "diagnostics": diagnostics,
        }

    @torch.no_grad()
    def update_ema(self, step: int | None = None) -> None:
        # Call after optimizer.step() so the EMA teacher lags the student CE path.
        momentum = self.ema_momentum_at_step(step)
        for ema_ce_norm, ema_compressor, block in zip(self.ema_ce_norms, self.ema_compressors, self.blocks):
            ema_ce_norm.weight.lerp_(block.ce_norm.weight, 1.0 - momentum)
            for ema_parameter, student_parameter in zip(ema_compressor.module.parameters(), block.compressor.parameters()):
                ema_parameter.lerp_(student_parameter, 1.0 - momentum)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        upgraded = state_dict.copy()
        for layer_index in range(len(self.blocks)):
            ema_norm_key = f"ema_ce_norms.{layer_index}.weight"
            student_norm_key = f"blocks.{layer_index}.ce_norm.weight"
            if ema_norm_key not in upgraded and student_norm_key in upgraded:
                upgraded[ema_norm_key] = upgraded[student_norm_key].detach().clone()

        return super().load_state_dict(upgraded, strict=strict, assign=assign)
