import torch

from text_jepa.models.tower import EncoderTower
from text_jepa.utils.ema import update_ema


def test_encoder_tower_returns_bld_shape():
    tower = EncoderTower(
        vocab_size=32,
        max_length=16,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        ffn_dim=32,
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)

    output = tower(input_ids, attention_mask=attention_mask)

    # The tower is just embeddings plus encoder, so it should preserve the sequence axis.
    assert output.shape == (2, 3, 8)


def test_encoder_tower_rejects_non_batched_input_ids():
    tower = EncoderTower(
        vocab_size=32,
        max_length=16,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
    )
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    try:
        tower(input_ids)
    except ValueError as exc:
        assert "input_ids must have shape (B, L)" in str(exc)
    else:
        raise AssertionError("Expected EncoderTower to reject non-batched input ids")


def test_encoder_tower_matches_embedding_then_encoder_contract():
    tower = EncoderTower(
        vocab_size=32,
        max_length=16,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        ffn_dim=32,
    )

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)

    manual_output = tower.encoder(
        tower.embeddings(input_ids),
        attention_mask=attention_mask,
    )
    tower_output = tower(input_ids, attention_mask=attention_mask)

    # This checks the tower adds no hidden behavior beyond composing its two submodules.
    assert torch.equal(tower_output, manual_output)


def test_update_ema_can_hard_sync_two_towers():
    context_tower = EncoderTower(
        vocab_size=32,
        max_length=16,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
    )
    target_tower = EncoderTower(
        vocab_size=32,
        max_length=16,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
    )

    # momentum=0.0 should make the target tower an exact copy of the context tower.
    update_ema(target_tower, context_tower, momentum=0.0)

    for target_parameter, context_parameter in zip(
        target_tower.parameters(),
        context_tower.parameters(),
    ):
        assert torch.equal(target_parameter, context_parameter)
