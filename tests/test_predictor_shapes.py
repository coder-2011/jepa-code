import torch

from text_jepa.models.predictor import Predictor


def make_predictor():
    return Predictor(
        hidden_dim=8,
        max_length=16,
        num_layers=2,
        num_heads=2,
        ffn_dim=32,
    )


def make_inputs():
    context_states = torch.randn(2, 5, 8)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
    target_positions = torch.tensor([[1, 3, 0], [0, 2, 4]], dtype=torch.long)
    target_valid_mask = torch.tensor([[True, True, False], [True, True, True]])
    return context_states, attention_mask, target_positions, target_valid_mask


def test_predictor_returns_btd_shape():
    predictor = make_predictor()
    context_states, attention_mask, target_positions, target_valid_mask = make_inputs()

    output = predictor(context_states, attention_mask, target_positions, target_valid_mask)

    # Predictor output is indexed by padded target slots, not full sequence length.
    assert output.shape == (2, 3, 8)


def test_predictor_accepts_padded_target_slots():
    predictor = make_predictor()
    context_states, attention_mask, target_positions, target_valid_mask = make_inputs()
    target_valid_mask[0, 2] = False

    output = predictor(context_states, attention_mask, target_positions, target_valid_mask)

    # Invalid target slots remain present in shape and are masked semantically rather than structurally removed.
    assert output.shape == (2, 3, 8)


def test_predictor_rejects_shape_mismatch_between_targets_and_valid_mask():
    predictor = make_predictor()
    context_states, attention_mask, target_positions, _ = make_inputs()
    target_valid_mask = torch.ones(2, 2, dtype=torch.bool)

    try:
        predictor(context_states, attention_mask, target_positions, target_valid_mask)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected Predictor to reject mismatched target padding shapes")


def test_predictor_rejects_out_of_range_target_positions():
    predictor = make_predictor()
    context_states, attention_mask, target_positions, target_valid_mask = make_inputs()
    target_positions[0, 1] = 16

    try:
        predictor(context_states, attention_mask, target_positions, target_valid_mask)
    except ValueError as exc:
        assert "within [0, max_length)" in str(exc)
    else:
        raise AssertionError("Expected Predictor to reject an out-of-range target position")


def test_predictor_output_depends_on_target_positions():
    predictor = make_predictor()
    context_states, attention_mask, target_positions, target_valid_mask = make_inputs()
    shifted_positions = target_positions.clone()
    shifted_positions[0, 1] = 2

    output_a = predictor(context_states, attention_mask, target_positions, target_valid_mask)
    output_b = predictor(context_states, attention_mask, shifted_positions, target_valid_mask)

    # Query position embeddings should make the same context produce different outputs at different slots.
    assert not torch.allclose(output_a, output_b)


def test_predictor_output_depends_on_context_states():
    predictor = make_predictor()
    context_states, attention_mask, target_positions, target_valid_mask = make_inputs()
    shifted_context = context_states.clone()
    shifted_context[0, 0, 0] += 1.0

    output_a = predictor(context_states, attention_mask, target_positions, target_valid_mask)
    output_b = predictor(shifted_context, attention_mask, target_positions, target_valid_mask)

    # Decoder cross-attention should make predictions responsive to context-side latent changes.
    assert not torch.allclose(output_a, output_b)
