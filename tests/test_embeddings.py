import torch

from text_jepa.models.embeddings import TextEmbeddings


def test_text_embeddings_returns_bld_shape():
    embeddings = TextEmbeddings(vocab_size=32, max_length=16, hidden_dim=8)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    output = embeddings(input_ids)

    # The embedding layer should turn integer token ids (B, L) into dense token vectors (B, L, D).
    assert output.shape == (2, 3, 8)


def test_text_embeddings_use_position_information():
    embeddings = TextEmbeddings(vocab_size=32, max_length=16, hidden_dim=8)
    input_ids = torch.tensor([[7, 7, 7]], dtype=torch.long)

    output = embeddings(input_ids)

    # Repeated token ids at different positions should still differ after adding position embeddings.
    assert not torch.equal(output[0, 0], output[0, 1])
