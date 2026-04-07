import torch
from torch import nn

from ._norms import make_rms_norm


class Predictor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        max_length,
        num_layers,
        num_heads,
        ffn_dim,
        dropout=0.0,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.query_seed = nn.Parameter(torch.zeros(hidden_dim))
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        layer.norm1 = make_rms_norm(hidden_dim)
        layer.norm2 = make_rms_norm(hidden_dim)
        layer.norm3 = make_rms_norm(hidden_dim)
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
        )
        self.final_norm = make_rms_norm(hidden_dim)

    def forward(self, context_states, attention_mask, target_positions, target_valid_mask):
        if context_states.ndim != 3:
            raise ValueError("context_states must have shape (B, L, D)")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape (B, L)")
        if target_positions.ndim != 2:
            raise ValueError("target_positions must have shape (B, T_max)")
        if target_valid_mask.ndim != 2:
            raise ValueError("target_valid_mask must have shape (B, T_max)")

        batch_size, sequence_length, hidden_dim = context_states.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError("context_states hidden dimension must match predictor hidden_dim")
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must match the first two dimensions of context_states")
        if target_positions.shape != target_valid_mask.shape:
            raise ValueError("target_positions and target_valid_mask must have the same shape")
        if sequence_length > self.max_length:
            raise ValueError("context_states sequence length exceeds the configured max_length")
        if torch.any(target_positions < 0) or torch.any(target_positions >= self.max_length):
            raise ValueError("target_positions must be within [0, max_length)")

        target_queries = self.position_embedding(target_positions)
        target_queries = target_queries + self.query_seed.view(1, 1, -1)

        memory_key_padding_mask = attention_mask == 0
        target_key_padding_mask = ~target_valid_mask.to(torch.bool)

        predicted_states = self.decoder(
            tgt=target_queries,
            memory=context_states,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.final_norm(predicted_states)
