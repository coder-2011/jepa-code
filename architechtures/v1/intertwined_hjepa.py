from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

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
    # Keep the v1 config narrow: only the knobs used by the pseudocode path.
    vocab_size: int
    max_length: int
    residual_dim: int
    compressed_dim: int
    depth: int
    num_heads: int = 1
    predictor_hidden_dim: int | None = None
    dropout: float = 0.0
    ema_momentum: float = 0.996
    lambda_jepa: float = 0.1
    jepa_warmup_steps: int = 0
    tie_weights: bool = True


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
        # x: (..., D) -> (..., K)
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
    # Keep the contract explicit here; shape mismatches otherwise fail later with noisy broadcast errors.
    if delta_l.shape != z_l.shape or delta_l.shape != target_z_l.shape:
        raise ValueError("delta_l, z_l, and target_z_l must have the same shape")
    if delta_l.ndim < 2:
        raise ValueError("delta_l, z_l, and target_z_l must have at least 2 dimensions")

    # Stop gradients through both pieces of the teacher delta target.
    if valid_mask is None:
        return F.mse_loss(delta_l, target_z_l.detach() - z_l.detach())

    if valid_mask.shape != delta_l.shape[:-1]:
        raise ValueError("valid_mask must match the leading dimensions of delta_l")
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
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, L, vocab_size)")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape (B, L) matching logits")
    if logits.shape[1] < 2:
        raise ValueError("next_token_loss requires sequence length >= 2")
    if valid_mask is not None and valid_mask.shape != labels.shape:
        raise ValueError("valid_mask must match labels shape")

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

        predictor_hidden_dim = config.predictor_hidden_dim or config.compressed_dim
        self.config = config
        self.ema_momentum = float(config.ema_momentum)

        # Plain learned token + position embeddings for v1.
        self.token_embedding = nn.Embedding(config.vocab_size, config.residual_dim)
        self.position_embedding = nn.Embedding(config.max_length, config.residual_dim)
        # Uniform blocks keep the stack simple; the last block still produces z/delta even though it has no JEPA loss.
        self.blocks = nn.ModuleList(
            [
                IntertwinedBlock(
                    residual_dim=config.residual_dim,
                    compressed_dim=config.compressed_dim,
                    predictor_hidden_dim=predictor_hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
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
        self.final_norm = nn.RMSNorm(config.residual_dim)
        # Tied case reuses the token embedding matrix directly in forward; untied case owns a separate head.
        self.lm_head = (
            None
            if config.tie_weights
            else nn.Linear(config.residual_dim, config.vocab_size, bias=False)
        )

    def student_parameters(self):
        # EMA parameters are frozen, so this naturally returns only trainable student weights.
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L) -> h_0: (B, L, D)
        batch_size, sequence_length = input_ids.shape
        # Keep positions explicit and learned for the first pass; no RoPE or cache-specific logic yet.
        position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, sequence_length)
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)

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
        # Project the final residual stream to logits, using the embedding matrix directly when tied.
        logit_weight = self.token_embedding.weight if self.lm_head is None else self.lm_head.weight
        logits = F.linear(self.final_norm(h), logit_weight)

        targets = []
        jepa_losses = []
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

        # Average JEPA equally across all non-final layers.
        loss_jepa = torch.stack(jepa_losses).mean()
        lambda_eff = warmup_weight(self.config.lambda_jepa, step, self.config.jepa_warmup_steps)
        loss_main = next_token_loss(logits, labels, valid_mask) if labels is not None else None
        loss = lambda_eff * loss_jepa if loss_main is None else loss_main + lambda_eff * loss_jepa

        # Keep a few cheap summaries around for smoke tests and debugging.
        diagnostics = {
            "lambda_jepa": lambda_eff,
            "loss_jepa_layers": [loss.detach() for loss in jepa_losses],
            "z_variance": [z.detach().float().var() for z in compressed],
            "delta_norm": [delta.detach().float().norm() for delta in deltas],
        }

        return {
            "loss": loss,
            "loss_main": loss_main,
            "loss_jepa": loss_jepa,
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
