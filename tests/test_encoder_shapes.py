import torch

from text_jepa.models.embeddings import TextEmbeddings
from text_jepa.models.encoder import Encoder


def test_encoder_preserves_bld_shape():
    encoder = Encoder(num_layers=2, hidden_dim=8, num_heads=2, ffn_dim=32)
    x = torch.randn(2, 5, 8)
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    output = encoder(x, attention_mask=attention_mask)

    assert output.shape == (2, 5, 8)


def test_encoder_accepts_padding_mask():
    encoder = Encoder(num_layers=1, hidden_dim=8, num_heads=2, ffn_dim=32)
    x = torch.randn(2, 5, 8)
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        dtype=torch.long,
    )

    output = encoder(x, attention_mask=attention_mask)

    assert output.shape == (2, 5, 8)


def test_embeddings_and_encoder_integrate_cleanly():
    embeddings = TextEmbeddings(vocab_size=32, max_length=16, hidden_dim=8)
    encoder = Encoder(num_layers=2, hidden_dim=8, num_heads=2, ffn_dim=32)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)

    embedded = embeddings(input_ids)
    output = encoder(embedded, attention_mask=attention_mask)

    assert output.shape == (2, 3, 8)


def test_encoder_rejects_head_dimension_mismatch():
    try:
        Encoder(num_layers=1, hidden_dim=10, num_heads=3, ffn_dim=32)
    except ValueError as exc:
        assert "divisible by num_heads" in str(exc)
    else:
        raise AssertionError("Expected Encoder to reject an incompatible head configuration")
