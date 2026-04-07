from torch import nn


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, ffn_dim, dropout=0.0):
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, attention_mask=None):
        if x.ndim != 3:
            raise ValueError("x must have shape (B, L, D)")

        src_key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("attention_mask must have shape (B, L)")
            if attention_mask.shape != x.shape[:2]:
                raise ValueError("attention_mask must match the first two dimensions of x")
            src_key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(x)
