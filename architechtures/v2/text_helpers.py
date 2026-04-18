import torch
import torch.nn.functional as F
from torch import nn


class HFTokenizer:
    """Thin wrapper around a Hugging Face tokenizer."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> "HFTokenizer":
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(name_or_path, **kwargs))

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    def encode(self, text: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, residual_dim: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, residual_dim)
        self.position_embedding = nn.Embedding(max_length, residual_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = input_ids.shape[1]
        assert sequence_length <= self.max_length, "sequence length exceeds max_length"
        position_ids = torch.arange(sequence_length, device=input_ids.device)[None, :]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class LMHead(nn.Module):
    def __init__(
        self,
        residual_dim: int,
        vocab_size: int,
        token_embedding: nn.Embedding,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.proj = None if tie_weights else nn.Linear(residual_dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.token_embedding.weight if self.proj is None else self.proj.weight
        return F.linear(hidden_states, weight)
