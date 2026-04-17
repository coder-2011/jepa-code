from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

from intertwined_hjepa import IntertwinedConfig, IntertwinedHJEPA, jepa_delta_loss
from sigreg import SlicedEppsPulleySIGReg

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
    except ValueError as exc:
        assert "valid_mask" in str(exc)
    else:
        raise AssertionError("Expected an empty valid_mask to be rejected")


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
    assert all(parameter.grad is None for parameter in model.final_block.parameters())
    assert all(parameter.grad is None for parameter in model.ema_compressors.parameters())
    assert all(parameter.grad is None for parameter in model.output_target_norm.parameters())
    assert all(parameter.grad is None for parameter in model.output_target_compressor.parameters())


def test_sliced_epps_pulley_sigreg_is_finite_and_differentiable():
    sigreg = SlicedEppsPulleySIGReg(num_slices=8, n_points=5)
    z = torch.randn(6, 4, requires_grad=True)

    loss = sigreg(z)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


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


def test_sigreg_loss_updates_compressor_only():
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
        assert any(parameter.grad is not None for parameter in block.compressor.parameters())
        assert all(parameter.grad is None for parameter in block.attn.parameters())
        assert all(parameter.grad is None for parameter in block.ce_norm.parameters())
        assert all(parameter.grad is None for parameter in block.predictor.parameters())
        assert all(parameter.grad is None for parameter in block.projector.parameters())

    assert all(parameter.grad is None for parameter in model.final_block.parameters())
    assert all(parameter.grad is None for parameter in model.embeddings.parameters())
    assert all(parameter.grad is None for parameter in model.ema_ce_norms.parameters())
    assert all(parameter.grad is None for parameter in model.ema_compressors.parameters())
    assert all(parameter.grad is None for parameter in model.output_target_norm.parameters())
    assert all(parameter.grad is None for parameter in model.output_target_compressor.parameters())


def test_training_step_updates_students_then_ema():
    model = make_model()
    optimizer = torch.optim.AdamW(model.student_parameters(), lr=0.01)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)

    student_before = [parameter.detach().clone() for parameter in model.student_parameters()]
    ema_before = [
        parameter.detach().clone()
        for parameter in list(model.ema_ce_norms.parameters()) + list(model.ema_compressors.parameters())
    ]
    output_target_before = [
        parameter.detach().clone()
        for parameter in list(model.output_target_norm.parameters())
        + list(model.output_target_compressor.parameters())
    ]

    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs["loss"].backward()

    assert any(parameter.grad is not None for parameter in model.blocks.parameters())
    assert any(parameter.grad is not None for parameter in model.final_block.parameters())
    assert all(parameter.grad is None for parameter in model.ema_ce_norms.parameters())
    assert all(parameter.grad is None for parameter in model.ema_compressors.parameters())
    assert all(parameter.grad is None for parameter in model.output_target_norm.parameters())
    assert all(parameter.grad is None for parameter in model.output_target_compressor.parameters())

    optimizer.step()
    model.update_ema()

    student_after = [parameter.detach() for parameter in model.student_parameters()]
    ema_after = [
        parameter.detach()
        for parameter in list(model.ema_ce_norms.parameters()) + list(model.ema_compressors.parameters())
    ]
    output_target_after = [
        parameter.detach()
        for parameter in list(model.output_target_norm.parameters())
        + list(model.output_target_compressor.parameters())
    ]

    assert any(not torch.equal(before, after) for before, after in zip(student_before, student_after))
    assert any(not torch.equal(before, after) for before, after in zip(ema_before, ema_after))
    assert all(torch.equal(before, after) for before, after in zip(output_target_before, output_target_after))
