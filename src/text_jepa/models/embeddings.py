import torch
from torch import nn


class TextEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, L)")

        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError("sequence length exceeds the configured max_length")

        # Build one position index per token slot, then broadcast it across the batch.
        position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, sequence_length)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        return token_embeddings + position_embeddings
