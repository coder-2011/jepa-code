import torch

from intertwined_hjepa import (
    IntertwinedConfig,
    IntertwinedHJEPA,
    jepa_delta_loss,
    next_token_loss,
)


def make_model():
    return IntertwinedHJEPA(
        IntertwinedConfig(
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
        )
    )


def test_jepa_delta_loss_detaches_target_delta():
    delta = torch.randn(2, 3, 4, requires_grad=True)
    z = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn(2, 3, 4, requires_grad=True)

    loss = jepa_delta_loss(delta, z, target)
    loss.backward()

    assert delta.grad is not None
    assert z.grad is None
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


def test_jepa_delta_loss_rejects_shape_mismatch():
    delta = torch.randn(2, 3, 4)
    z = torch.randn(2, 3, 5)
    target = torch.randn(2, 3, 4)

    try:
        jepa_delta_loss(delta, z, target)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected mismatched JEPA tensors to be rejected")


def test_jepa_delta_loss_rejects_bad_valid_mask_shape():
    delta = torch.randn(2, 3, 4)
    z = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)
    valid_mask = torch.ones(2, 4, dtype=torch.bool)

    try:
        jepa_delta_loss(delta, z, target, valid_mask=valid_mask)
    except ValueError as exc:
        assert "valid_mask" in str(exc)
    else:
        raise AssertionError("Expected bad valid_mask shape to be rejected")


def test_next_token_loss_rejects_length_one_sequence():
    logits = torch.randn(2, 1, 16)
    labels = torch.zeros(2, 1, dtype=torch.long)

    try:
        next_token_loss(logits, labels)
    except ValueError as exc:
        assert "sequence length" in str(exc)
    else:
        raise AssertionError("Expected L=1 next-token loss to be rejected")


def test_ema_targets_have_no_gradients():
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert all(not target.requires_grad for target in outputs["targets"])


def test_training_step_updates_students_then_ema():
    model = make_model()
    optimizer = torch.optim.AdamW(model.student_parameters(), lr=0.01)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)

    student_before = [parameter.detach().clone() for parameter in model.student_parameters()]
    ema_before = [parameter.detach().clone() for parameter in model.ema_compressors.parameters()]

    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs["loss"].backward()

    assert any(parameter.grad is not None for parameter in model.blocks.parameters())
    assert all(parameter.grad is None for parameter in model.ema_compressors.parameters())

    optimizer.step()
    model.update_ema()

    student_after = [parameter.detach() for parameter in model.student_parameters()]
    ema_after = [parameter.detach() for parameter in model.ema_compressors.parameters()]

    assert any(not torch.equal(before, after) for before, after in zip(student_before, student_after))
    assert any(not torch.equal(before, after) for before, after in zip(ema_before, ema_after))
