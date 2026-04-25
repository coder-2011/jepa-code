from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent

from intertwined_hjepa import (
    IntertwinedConfig,
    IntertwinedHJEPA,
    auxiliary_layer_indices,
    jepa_delta_loss,
    jepa_state_loss,
    next_token_loss,
    rms_normalize_last_dim,
    split_scalars,
)
from sigreg import SIGReg

YAML_CONFIG = IntertwinedConfig.from_yaml(ROOT / "intertwined_hjepa.yaml")


def make_model():
    return IntertwinedHJEPA(
        replace(
            YAML_CONFIG,
            vocab_size=32,
            max_length=8,
            residual_dim=8,
            compressed_dim=4,
            depth=3,
            num_heads=2,
            predictor_hidden_dim=16,
            dropout=0.0,
            ema_momentum=0.5,
            lambda_jepa=0.1,
            beta_sigreg=0.0,
            sigreg_num_slices=8,
            sigreg_n_points=5,
        )
    )


def test_jepa_delta_loss_stops_teacher_but_updates_z():
    delta = torch.randn(2, 3, 4, requires_grad=True)
    z = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn(2, 3, 4, requires_grad=True)

    loss = jepa_delta_loss(delta, z, target)
    loss.backward()

    assert delta.grad is not None
    assert z.grad is not None
    assert target.grad is None


def test_jepa_delta_loss_rejects_empty_valid_mask():
    delta = torch.randn(2, 3, 4)
    z = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)
    valid_mask = torch.zeros(2, 3, dtype=torch.bool)

    try:
        jepa_delta_loss(delta, z, target, valid_mask=valid_mask)
    except AssertionError as exc:
        assert "valid_mask" in str(exc)
    else:
        raise AssertionError("Expected an empty valid_mask to be rejected")


def test_jepa_delta_loss_uses_normalized_latents():
    delta = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
    z = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    target = torch.tensor([[[3.0, 0.0], [0.0, 6.0]]])

    normalized_delta = rms_normalize_last_dim(delta)
    normalized_target_delta = rms_normalize_last_dim(target) - rms_normalize_last_dim(z)
    expected = torch.nn.functional.mse_loss(normalized_delta, normalized_target_delta)

    assert torch.allclose(jepa_delta_loss(delta, z, target), expected)


def test_jepa_state_loss_aligns_student_state_to_future_target_directly():
    z = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    target = torch.tensor([[[3.0, 0.0], [2.0, 0.0]]], requires_grad=True)

    expected = torch.nn.functional.mse_loss(rms_normalize_last_dim(z), rms_normalize_last_dim(target.detach()))
    loss = jepa_state_loss(z, target)
    loss.backward()

    assert torch.allclose(loss, expected)
    assert z.grad is not None
    assert target.grad is None


def test_next_token_loss_uses_already_shifted_labels_without_second_shift():
    labels = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
    logits = torch.full((1, 4, 20), -100.0)
    for index, token_id in enumerate(labels[0].tolist()):
        logits[0, index, token_id] = 100.0

    loss = next_token_loss(logits, labels)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_next_token_loss_masks_current_positions_directly():
    labels = torch.tensor([[3, 5, 7]], dtype=torch.long)
    logits = torch.full((1, 3, 10), -100.0)
    logits[0, 0, 3] = 100.0
    logits[0, 1, 0] = 100.0
    logits[0, 2, 7] = 100.0
    valid_mask = torch.tensor([[True, False, True]], dtype=torch.bool)

    loss = next_token_loss(logits, labels, valid_mask=valid_mask)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_next_token_jepa_masks_out_final_position():
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)

    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs["jepa_valid_mask"][:, :-1].all()
    assert not outputs["jepa_valid_mask"][:, -1].any()


def test_next_token_jepa_combines_sequence_valid_mask_with_tail_mask():
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    valid_mask = torch.tensor(
        [
            [True, False, True, True],
            [False, True, True, True],
        ],
        dtype=torch.bool,
    )

    outputs = model(input_ids=input_ids, labels=input_ids, valid_mask=valid_mask, compute_aux_losses=False)

    expected = valid_mask.clone()
    expected[:, -1] = False
    assert torch.equal(outputs["jepa_valid_mask"], expected)


def test_ema_targets_have_no_gradients():
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert all(not target.requires_grad for target in outputs["targets"])


def test_total_loss_sums_lm_and_local_jepa_losses():
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)

    expected = outputs["loss_main"] + model.config.lambda_jepa * torch.stack(outputs["loss_jepa_layers"]).sum()
    assert torch.allclose(outputs["loss"], expected)


def test_jepa_loss_updates_ce_path():
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)

    outputs = model(input_ids=input_ids)
    outputs["loss"].backward()

    first_block = model.blocks[0]
    assert any(parameter.grad is not None for parameter in first_block.attn.parameters())
    assert any(parameter.grad is not None for parameter in first_block.ce_norm.parameters())
    assert any(parameter.grad is not None for parameter in first_block.compressor.parameters())
    assert any(parameter.grad is not None for parameter in first_block.predictor.parameters())
    assert any(parameter.grad is not None for parameter in first_block.transition_mlp.parameters())
    assert all(parameter.grad is None for parameter in model.final_block.parameters())
    assert all(parameter.grad is None for parameter in model.ema_target_encoders.parameters())


def test_sliced_epps_pulley_sigreg_is_finite_and_differentiable():
    sigreg = SIGReg(num_slices=8, knots=5)
    z = torch.randn(6, 4, requires_grad=True)

    loss = sigreg(z)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_stacked_auxiliary_losses_match_individual_layer_losses():
    torch.manual_seed(123)
    model = IntertwinedHJEPA(
        replace(
            YAML_CONFIG,
            vocab_size=32,
            max_length=8,
            residual_dim=16,
            compressed_dim=8,
            depth=4,
            num_heads=4,
            predictor_hidden_dim=32,
            dropout=0.0,
            ema_momentum=0.5,
            lambda_jepa=0.1,
            beta_sigreg=0.05,
            sigreg_num_slices=8,
            sigreg_n_points=5,
        )
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]], dtype=torch.long)
    valid_mask = torch.tensor([[True, True, True, True, True, True], [True, True, True, False, False, False]])

    torch.manual_seed(999)
    outputs = model(input_ids=input_ids, labels=input_ids, valid_mask=valid_mask)

    individual_jepa_losses = [
        jepa_delta_loss(
            outputs["deltas"][index][:, :-1],
            outputs["z"][index][:, :-1],
            outputs["targets"][index][:, 1:],
            valid_mask=outputs["jepa_valid_mask"][:, :-1],
        )
        for index in range(len(model.blocks))
    ]

    sigreg_input = rms_normalize_last_dim(torch.stack(outputs["z"]))
    torch.manual_seed(999)
    individual_sigreg_losses = [
        model.sigreg(sigreg_input[index], sample_mask=valid_mask)
        for index in range(len(model.blocks))
    ]

    assert torch.allclose(torch.stack(outputs["loss_jepa_layers"]), torch.stack(individual_jepa_losses))
    assert torch.allclose(torch.stack(outputs["loss_sigreg_layers"]), torch.stack(individual_sigreg_losses))


def test_auxiliary_layer_indices_use_start_and_stride():
    assert auxiliary_layer_indices(num_layers=9, start=0, stride=1) == tuple(range(9))
    assert auxiliary_layer_indices(num_layers=9, start=6, stride=1) == (6, 7, 8)
    assert auxiliary_layer_indices(num_layers=9, start=3, stride=2) == (3, 5, 7)

    with pytest.raises(AssertionError, match="start"):
        auxiliary_layer_indices(num_layers=3, start=4, stride=1)
    with pytest.raises(AssertionError, match="stride"):
        auxiliary_layer_indices(num_layers=3, start=0, stride=0)


def test_auxiliary_losses_can_start_and_stride_across_layers():
    torch.manual_seed(123)
    model = IntertwinedHJEPA(
        replace(
            YAML_CONFIG,
            vocab_size=32,
            max_length=8,
            residual_dim=16,
            compressed_dim=8,
            depth=5,
            num_heads=4,
            predictor_hidden_dim=32,
            dropout=0.0,
            ema_momentum=0.5,
            lambda_jepa=0.1,
            beta_sigreg=0.05,
            sigreg_num_slices=8,
            sigreg_n_points=5,
            auxiliary_layer_start=1,
            auxiliary_layer_stride=2,
        )
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]], dtype=torch.long)
    valid_mask = torch.tensor([[True, True, True, True, True, True], [True, True, True, False, False, False]])

    outputs = model(input_ids=input_ids, labels=input_ids, valid_mask=valid_mask)

    assert model.auxiliary_layer_indices == (1, 3)
    assert torch.equal(outputs["loss_jepa_layers"][0], torch.zeros_like(outputs["loss_jepa_layers"][0]))
    assert outputs["loss_jepa_layers"][1] > 0
    assert torch.equal(outputs["loss_jepa_layers"][2], torch.zeros_like(outputs["loss_jepa_layers"][2]))
    assert outputs["loss_jepa_layers"][3] > 0
    assert torch.equal(outputs["loss_sigreg_layers"][0], torch.zeros_like(outputs["loss_sigreg_layers"][0]))
    assert outputs["loss_sigreg_layers"][1] > 0
    assert torch.equal(outputs["loss_sigreg_layers"][2], torch.zeros_like(outputs["loss_sigreg_layers"][2]))
    assert outputs["loss_sigreg_layers"][3] > 0
    assert torch.equal(outputs["loss_jepa"], torch.stack(outputs["loss_jepa_layers"]).sum())
    assert torch.equal(outputs["loss_sigreg"], torch.stack(outputs["loss_sigreg_layers"]).sum())


def test_auxiliary_target_groups_can_use_direct_state_prediction():
    torch.manual_seed(123)
    model = IntertwinedHJEPA(
        replace(
            YAML_CONFIG,
            vocab_size=32,
            max_length=8,
            residual_dim=16,
            compressed_dim=8,
            depth=5,
            num_heads=4,
            predictor_hidden_dim=32,
            dropout=0.0,
            ema_momentum=0.5,
            lambda_jepa=0.1,
            beta_sigreg=0.0,
            sigreg_num_slices=8,
            sigreg_n_points=5,
            auxiliary_target_groups=[
                {"layers": [2, 3], "target_type": "final_layer", "prediction": "state", "horizon": [2, 4]},
            ],
        )
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.long)

    outputs = model(input_ids=input_ids, labels=input_ids)

    assert [(spec.layer_index, spec.target_type, spec.prediction_type, spec.horizon_start, spec.horizon_end) for spec in model.auxiliary_target_specs] == [
        (2, "final_layer", "state", 2, 4),
        (3, "final_layer", "state", 2, 4),
    ]
    assert outputs["diagnostics"]["auxiliary_prediction_type"] == ["state", "state"]
    assert [index for index, loss in enumerate(outputs["loss_jepa_layers"]) if loss.item() > 0.0] == [2, 3]
    assert torch.equal(outputs["loss_jepa_layers"][0], torch.zeros_like(outputs["loss_jepa_layers"][0]))

    spec = model.auxiliary_target_specs[0]
    target_sequence = model.compute_raw_jepa_target_for_layer(
        spec.layer_index,
        outputs["post_attn_states"],
        final_states=outputs["final_states"],
        target_type=spec.target_type,
    )
    assert not torch.allclose(target_sequence, outputs["targets"][0])

    outputs["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.blocks[2].compressor.parameters())
    assert all(parameter.grad is None for parameter in model.blocks[2].predictor.parameters())
    assert all(parameter.grad is None for parameter in model.ema_target_encoders.parameters())


def test_total_loss_includes_local_sigreg_when_enabled():
    model = IntertwinedHJEPA(
        replace(
            YAML_CONFIG,
            vocab_size=32,
            max_length=8,
            residual_dim=8,
            compressed_dim=4,
            depth=3,
            num_heads=2,
            predictor_hidden_dim=16,
            dropout=0.0,
            ema_momentum=0.5,
            lambda_jepa=0.1,
            beta_sigreg=0.05,
            sigreg_num_slices=8,
            sigreg_n_points=5,
        )
    )
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)

    expected = (
        outputs["loss_main"]
        + model.config.lambda_jepa * torch.stack(outputs["loss_jepa_layers"]).sum()
        + model.config.beta_sigreg * torch.stack(outputs["loss_sigreg_layers"]).sum()
    )
    assert torch.allclose(outputs["loss"], expected)
    assert outputs["loss_sigreg"] > 0


def test_forward_can_skip_auxiliary_losses_and_use_only_lm_loss():
    model = IntertwinedHJEPA(
        replace(
            YAML_CONFIG,
            vocab_size=32,
            max_length=8,
            residual_dim=8,
            compressed_dim=4,
            depth=3,
            num_heads=2,
            predictor_hidden_dim=16,
            dropout=0.0,
            ema_momentum=0.5,
            lambda_jepa=0.1,
            beta_sigreg=0.05,
            sigreg_num_slices=8,
            sigreg_n_points=5,
        )
    )
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)

    outputs = model(input_ids=input_ids, labels=input_ids, compute_aux_losses=False)

    assert torch.equal(outputs["loss"], outputs["loss_main"])
    assert torch.equal(outputs["loss_jepa"], torch.zeros_like(outputs["loss_jepa"]))
    assert torch.equal(outputs["loss_sigreg"], torch.zeros_like(outputs["loss_sigreg"]))
    assert len(outputs["targets"]) == len(model.blocks)
    assert all(torch.equal(loss, torch.zeros_like(loss)) for loss in outputs["loss_jepa_layers"])
    assert all(torch.equal(loss, torch.zeros_like(loss)) for loss in outputs["loss_sigreg_layers"])
    assert outputs["diagnostics"]["compute_aux_losses"] is False


def test_sigreg_loss_updates_encoder_path():
    model = IntertwinedHJEPA(
        replace(
            YAML_CONFIG,
            vocab_size=32,
            max_length=8,
            residual_dim=8,
            compressed_dim=4,
            depth=3,
            num_heads=2,
            predictor_hidden_dim=16,
            dropout=0.0,
            ema_momentum=0.5,
            lambda_jepa=0.0,
            beta_sigreg=1.0,
            sigreg_num_slices=8,
            sigreg_n_points=5,
        )
    )
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)

    outputs = model(input_ids=input_ids)
    outputs["loss"].backward()

    for block in model.blocks:
        assert any(parameter.grad is not None for parameter in block.attn.parameters())
        assert any(parameter.grad is not None for parameter in block.ce_norm.parameters())
        assert any(parameter.grad is not None for parameter in block.compressor.parameters())

    assert any(parameter.grad is not None for parameter in model.blocks[0].transition_mlp.parameters())
    assert all(parameter.grad is None for parameter in model.blocks[-1].transition_mlp.parameters())

    assert all(parameter.grad is None for parameter in model.final_block.parameters())
    assert any(parameter.grad is not None for parameter in model.embeddings.parameters())
    assert all(parameter.grad is None for parameter in model.ema_target_encoders.parameters())




def test_training_step_updates_students_then_ema():
    model = make_model()
    optimizer = torch.optim.AdamW(model.student_parameters(), lr=0.01)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)

    student_before = [parameter.detach().clone() for parameter in model.student_parameters()]
    ema_before = [parameter.detach().clone() for parameter in model.ema_target_encoders.parameters()]

    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs["loss"].backward()

    assert any(parameter.grad is not None for parameter in model.blocks.parameters())
    assert any(parameter.grad is not None for parameter in model.final_block.parameters())
    assert all(parameter.grad is None for parameter in model.ema_target_encoders.parameters())

    optimizer.step()
    model.update_ema()

    student_after = [parameter.detach() for parameter in model.student_parameters()]
    ema_after = [parameter.detach() for parameter in model.ema_target_encoders.parameters()]

    assert any(not torch.equal(before, after) for before, after in zip(student_before, student_after))
    assert any(not torch.equal(before, after) for before, after in zip(ema_before, ema_after))


def test_scheduled_ema_update_uses_step():
    model = IntertwinedHJEPA(
        replace(
            YAML_CONFIG,
            vocab_size=32,
            max_length=8,
            residual_dim=8,
            compressed_dim=4,
            depth=3,
            num_heads=2,
            predictor_hidden_dim=16,
            dropout=0.0,
            ema_momentum=0.0,
            ema_momentum_final=1.0,
            ema_warmup_steps=10,
            lambda_jepa=0.1,
            beta_sigreg=0.0,
            sigreg_num_slices=8,
            sigreg_n_points=5,
        )
    )

    block = model.blocks[0]
    ema_norm = model.ema_target_encoders[0].ce_norm
    ema_before = ema_norm.weight.detach().clone()

    with torch.no_grad():
        block.ce_norm.weight.fill_(2.0)

    assert model.ema_momentum_at_step(0) == pytest.approx(0.1)
    assert model.ema_momentum_at_step(9) == pytest.approx(1.0)

    model.update_ema(step=0)

    expected = ema_before.lerp(block.ce_norm.weight, 0.9)
    assert torch.allclose(ema_norm.weight, expected)


def test_split_scalars_materializes_zero_dim_tensors():
    values = torch.tensor([1.0, 2.0, 3.0])

    pieces = split_scalars(values)

    assert len(pieces) == 3
    assert all(piece.shape == torch.Size([]) for piece in pieces)
    values[0] = 9.0
    assert pieces[0].item() == pytest.approx(1.0)
