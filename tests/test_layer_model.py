import torch

from text_jepa.models.layer_model import LayerModel
from text_jepa.utils.ema import update_ema


def make_layer_model():
    return LayerModel(
        vocab_size=32,
        max_length=16,
        hidden_dim=8,
        encoder_num_layers=2,
        encoder_num_heads=2,
        encoder_ffn_dim=32,
        predictor_num_layers=2,
        predictor_num_heads=2,
        predictor_ffn_dim=32,
        ema_momentum=0.996,
    )


def make_inputs():
    return {
        "input_ids_full": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long),
        "input_ids_ctx": torch.tensor([[1, 9, 3, 4], [5, 6, 9, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long),
        "target_positions": torch.tensor([[1, 3], [2, 0]], dtype=torch.long),
        "target_valid_mask": torch.tensor([[True, False], [True, False]]),
    }


def test_layer_model_forward_returns_expected_shapes():
    model = make_layer_model()
    outputs = model(**make_inputs())

    # The model returns both dense tower states and gathered target-side supervision tensors.
    assert outputs["context_states"].shape == (2, 4, 8)
    assert outputs["target_states"].shape == (2, 4, 8)
    assert outputs["predicted_target_states"].shape == (2, 2, 8)
    assert outputs["target_target_states"].shape == (2, 2, 8)
    assert outputs["target_valid_mask"].shape == (2, 2)
    assert outputs["loss"].ndim == 0


def test_layer_model_forward_returns_expected_keys():
    model = make_layer_model()
    outputs = model(**make_inputs())

    # Keep the public forward payload stable because training and debugging code both consume it.
    assert set(outputs) == {
        "context_states",
        "target_states",
        "predicted_target_states",
        "target_target_states",
        "target_valid_mask",
        "loss",
    }


def test_layer_model_blocks_gradients_to_target_tower():
    model = make_layer_model()
    outputs = model(**make_inputs())

    outputs["loss"].backward()

    # Only the student path should receive gradients; the teacher stays stop-gradient.
    assert any(parameter.grad is not None for parameter in model.context_tower.parameters())
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target_tower.parameters())


def test_layer_model_target_tower_can_be_updated_by_ema():
    model = make_layer_model()
    first_context_parameter = next(model.context_tower.parameters())
    first_target_parameter = next(model.target_tower.parameters())

    with torch.no_grad():
        first_context_parameter.add_(1.0)
        target_before = first_target_parameter.clone()

    # EMA should move the teacher even though it never receives optimizer gradients directly.
    update_ema(model.target_tower, model.context_tower, model.ema_momentum)

    assert not torch.equal(first_target_parameter, target_before)
