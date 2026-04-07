from torch import nn

from .embeddings import TextEmbeddings
from .encoder import Encoder


class EncoderTower(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_length,
        hidden_dim,
        num_layers,
        num_heads,
        ffn_dim,
        dropout=0.0,
    ):
        super().__init__()
        # Keep the tower as a thin composition layer so context and target towers stay structurally identical.
        self.embeddings = TextEmbeddings(vocab_size, max_length, hidden_dim)
        self.encoder = Encoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, L)")
        # Tower output remains a full-sequence latent grid for every token position.
        x = self.embeddings(input_ids)
        return self.encoder(x, attention_mask=attention_mask)
