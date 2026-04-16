from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from sigreg import SlicedEppsPulleySIGReg
from text_helpers import LMHead, TokenEmbeddings

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
    beta_sigreg: float
    sigreg_warmup_steps: int
    sigreg_num_slices: int
    sigreg_t_max: float
    sigreg_n_points: int
    tie_weights: bool

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


class DeltaPredictor(nn.Module):
    def __init__(self, compressed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # Predict the "one-layer-future" delta directly in compressed space.
        self.net = nn.Sequential(
            nn.RMSNorm(compressed_dim),
            nn.Linear(compressed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, compressed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., K) -> (..., K)
        return self.net(x)


def init_intertwined_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.RMSNorm) and module.weight is not None:
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)


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
        self.attn_norm = nn.RMSNorm(residual_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=residual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        # Compressor path: D -> K, predictor path: K -> K, projector path: K -> D.
        self.ce_norm = nn.RMSNorm(residual_dim)
        self.compressor = SimpleCompressor(residual_dim, compressed_dim, dropout=dropout)
        self.predictor = DeltaPredictor(compressed_dim, predictor_hidden_dim, dropout=dropout)
        self.projector = nn.Sequential(
            nn.RMSNorm(compressed_dim),
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
        sequence_length = x_l_normed.shape[1]
        # The JEPA block also carries the LM head, so attention must stay causal.
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=x_l_normed.device,
        ).triu(1)
        attn_out, _ = self.attn(
            x_l_normed,
            x_l_normed,
            x_l_normed,
            attn_mask=causal_mask,
            need_weights=False,
        )
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


def jepa_delta_loss(
    delta_l: torch.Tensor,
    z_l: torch.Tensor,
    target_z_l: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Keep the JEPA contract exact: no silent broadcasting across batch, sequence, or compressed width.
    if delta_l.shape != z_l.shape or delta_l.shape != target_z_l.shape:
        raise ValueError("delta_l, z_l, and target_z_l must have exactly the same shape")

    # Stop gradients through both pieces of the teacher delta target.
    if valid_mask is None:
        return F.mse_loss(delta_l, target_z_l.detach() - z_l.detach())

    assert valid_mask.shape == delta_l.shape[:-1], "valid_mask must match the leading shape of delta_l"
    if not valid_mask.any():
        raise ValueError("valid_mask selects no JEPA loss positions")

    # valid_mask is over token positions (B, L); broadcast it across the compressed width K.
    error = F.mse_loss(delta_l, target_z_l.detach() - z_l.detach(), reduction="none")
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
    h_l_post_attn = h_l + Attention_l(RMSNorm(h_l))
    z_l = CE_l(RMSNorm(h_l_post_attn))
    d_l = Pred_l(z_l)
    h_{l+1} = h_l_post_attn + Proj_l(z_l + d_l)

    target_z_l = sg(CEbar_{l+1}(RMSNorm(h_{l+1}_post_attn)))
    L_jepa_l = MSE(d_l, target_z_l - sg(z_l))
    """

    def __init__(self, config: IntertwinedConfig):
        super().__init__()
        assert config.depth >= 2, "depth must be at least 2"

        self.config = config
        self.ema_momentum = float(config.ema_momentum)

        # Plain learned token + position embeddings for v1.
        self.embeddings = TokenEmbeddings(
            vocab_size=config.vocab_size,
            max_length=config.max_length,
            residual_dim=config.residual_dim,
        )
        # Uniform blocks keep the stack simple; the last block still produces z/delta even though it has no JEPA loss.
        self.blocks = nn.ModuleList(
            [
                IntertwinedBlock(
                    residual_dim=config.residual_dim,
                    compressed_dim=config.compressed_dim,
                    predictor_hidden_dim=config.predictor_hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.final_norm = nn.RMSNorm(config.residual_dim)
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
        self.apply(init_intertwined_weights)
        # Track EMA copies only for the compressors, matching the architecture docs.
        ema_avg_fn = get_ema_multi_avg_fn(self.ema_momentum)
        self.ema_compressors = nn.ModuleList(
            AveragedModel(block.compressor, multi_avg_fn=ema_avg_fn, use_buffers=False)
            for block in self.blocks
        )
        for ema_compressor in self.ema_compressors:
            ema_compressor.requires_grad_(False)
            ema_compressor.eval()
            # AveragedModel already starts as a copy; setting n_averaged avoids the first update turning
            # into another hard copy instead of the EMA blend we want after optimizer.step().
            ema_compressor.n_averaged.fill_(1)

    def student_parameters(self):
        # EMA parameters are frozen, so this naturally returns only trainable student weights.
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L) -> h_0: (B, L, D)
        return self.embeddings(input_ids)

    @torch.no_grad()
    def compute_jepa_target_for_layer(
        self,
        layer_index: int,
        post_attn_states: list[torch.Tensor],
    ) -> torch.Tensor:
        # For layer l, the teacher target comes from the next layer's post-attention state:
        # post_attn_states[layer_index + 1]: (B, L, D) -> target_z_l: (B, L, K)
        next_block = self.blocks[layer_index + 1]
        next_post_attn = post_attn_states[layer_index + 1]
        # Mirror the student's CE input path, but swap in the EMA compressor from layer l+1.
        target_z_l = self.ema_compressors[layer_index + 1].module(
            next_block.ce_norm(next_post_attn)
        )
        return target_z_l.detach()

    def compute_sigreg_input_for_layer(
        self,
        layer_index: int,
        post_attn_states: list[torch.Tensor],
    ) -> torch.Tensor:
        # SIGReg regularizes the layer representation directly, so gradients are
        # allowed to flow through the post-attention state and earlier computation.
        block = self.blocks[layer_index]
        return block.compressor(block.ce_norm(post_attn_states[layer_index]))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        step: int | None = None,
    ) -> dict[str, object]:
        """
        Args:
            input_ids:  (B, L)
            labels:     (B, L), optional next-token labels
            valid_mask: (B, L), optional loss mask on token positions

        Returns:
            logits:            (B, L, vocab_size)
            final_states:      (B, L, D)
            states:            depth + 1 residual states, each (B, L, D)
            post_attn_states:  depth states, each (B, L, D)
            z:                 depth compressed states, each (B, L, K)
            deltas:            depth predicted deltas, each (B, L, K)
            targets:           depth - 1 EMA targets, each (B, L, K)
            loss_sigreg_layers: depth - 1 local SIGReg losses
        """
        # h starts as h_0: dense token states of shape (B, L, D).
        h = self.embed(input_ids)
        states = []
        post_attn_states = []
        compressed = []
        deltas = []

        # Each block preserves the dense (B, L, *) leading shape; only the trailing width flips between D and K.
        for block in self.blocks:
            states.append(h)
            out = block.forward_student(h)
            h = out["x_next"]
            post_attn_states.append(out["x_post_attn"])
            compressed.append(out["z"])
            deltas.append(out["delta"])

        states.append(h)
        # Final norm stays in the model; the helper only does the D -> vocab projection.
        logits = self.lm_head(self.final_norm(h))

        targets = []
        jepa_losses = []
        sigreg_losses = []
        # Only layers 0..depth-2 receive JEPA loss because the final layer has no l+1 teacher target.
        for layer_index in range(self.config.depth - 1):
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
                sigreg_input_l = self.compute_sigreg_input_for_layer(layer_index, post_attn_states)
                if valid_mask is not None:
                    assert valid_mask.shape == sigreg_input_l.shape[:-1], (
                        "valid_mask must match the leading shape of SIGReg inputs"
                    )
                    sigreg_input_l = sigreg_input_l[valid_mask.to(torch.bool)]
                sigreg_losses.append(self.sigreg(sigreg_input_l))

        # Keep per-layer JEPA losses explicit and sum them into the total objective.
        loss_jepa = torch.stack(jepa_losses).sum()
        if not sigreg_losses:
            sigreg_losses = [logits.new_zeros(()) for _ in range(self.config.depth - 1)]
        loss_sigreg = torch.stack(sigreg_losses).sum()
        lambda_eff = warmup_weight(self.config.lambda_jepa, step, self.config.jepa_warmup_steps)
        beta_eff = warmup_weight(self.config.beta_sigreg, step, self.config.sigreg_warmup_steps)
        loss_main = next_token_loss(logits, labels, valid_mask) if labels is not None else None
        loss = logits.new_zeros(())
        if loss_main is not None:
            loss = loss + loss_main
        if lambda_eff != 0:
            loss = loss + lambda_eff * loss_jepa
        if beta_eff != 0:
            loss = loss + beta_eff * loss_sigreg

        # Keep a few cheap summaries around for smoke tests and debugging.
        z_stds = [z.detach().float().reshape(-1, z.shape[-1]).std(dim=0, unbiased=False) for z in compressed]
        diagnostics = {
            "lambda_jepa": lambda_eff,
            "beta_sigreg": beta_eff,
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
    def update_ema(self) -> None:
        # Call after optimizer.step() so the EMA teacher lags the student compressors.
        for ema_compressor, block in zip(self.ema_compressors, self.blocks):
            ema_compressor.update_parameters(block.compressor)
