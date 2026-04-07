import torch
from torch import nn

from ..losses.latent_loss import gather_target_states, masked_latent_mse
from ..utils.ema import update_ema
from .predictor import Predictor
from .tower import EncoderTower


class LayerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_length,
        hidden_dim,
        encoder_num_layers,
        encoder_num_heads,
        encoder_ffn_dim,
        predictor_num_layers,
        predictor_num_heads,
        predictor_ffn_dim,
        dropout=0.0,
        ema_momentum=0.996,
    ):
        super().__init__()

        self.ema_momentum = float(ema_momentum)
        self.context_tower = EncoderTower(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_dim=hidden_dim,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            ffn_dim=encoder_ffn_dim,
            dropout=dropout,
        )
        self.target_tower = EncoderTower(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_dim=hidden_dim,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            ffn_dim=encoder_ffn_dim,
            dropout=dropout,
        )
        self.predictor = Predictor(
            hidden_dim=hidden_dim,
            max_length=max_length,
            num_layers=predictor_num_layers,
            num_heads=predictor_num_heads,
            ffn_dim=predictor_ffn_dim,
            dropout=dropout,
        )
        update_ema(self.target_tower, self.context_tower, momentum=0.0)

    def forward(
        self,
        input_ids_full,
        input_ids_ctx,
        attention_mask,
        target_positions,
        target_valid_mask,
        target_mask=None,
        target_token_ids=None,
    ):
        for name, tensor, shape_name in (
            ("input_ids_full", input_ids_full, "(B, L)"),
            ("input_ids_ctx", input_ids_ctx, "(B, L)"),
            ("attention_mask", attention_mask, "(B, L)"),
            ("target_positions", target_positions, "(B, T_max)"),
            ("target_valid_mask", target_valid_mask, "(B, T_max)"),
        ):
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape {shape_name}")
        if input_ids_full.shape != input_ids_ctx.shape or input_ids_full.shape != attention_mask.shape:
            raise ValueError("input_ids_full, input_ids_ctx, and attention_mask must have the same shape")
        if target_positions.shape != target_valid_mask.shape:
            raise ValueError("target_positions and target_valid_mask must have the same shape")
        if input_ids_full.shape[0] != target_positions.shape[0]:
            raise ValueError("batch size must match between sequence inputs and target inputs")

        context_states = self.context_tower(input_ids_ctx, attention_mask=attention_mask)
        with torch.no_grad():
            target_states = self.target_tower(input_ids_full, attention_mask=attention_mask)

        predicted_target_states = self.predictor(
            context_states,
            attention_mask,
            target_positions,
            target_valid_mask,
        )
        target_target_states = gather_target_states(target_states, target_positions)
        loss = masked_latent_mse(
            predicted_target_states,
            target_target_states,
            target_valid_mask,
        )

        return {
            "context_states": context_states,
            "target_states": target_states,
            "predicted_target_states": predicted_target_states,
            "target_target_states": target_target_states,
            "target_valid_mask": target_valid_mask,
            "loss": loss,
        }
