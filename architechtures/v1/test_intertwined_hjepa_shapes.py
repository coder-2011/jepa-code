import torch

from intertwined_hjepa import IntertwinedBlock, IntertwinedConfig, IntertwinedHJEPA


def make_config():
    return IntertwinedConfig(
        vocab_size=32,
        max_length=8,
        residual_dim=8,
        compressed_dim=4,
        depth=3,
        num_heads=2,
        predictor_hidden_dim=16,
        dropout=0.0,
        ema_momentum=0.5,
        jepa_warmup_steps=0,
    )


def test_block_student_forward_shapes():
    block = IntertwinedBlock(
        residual_dim=8,
        compressed_dim=4,
        predictor_hidden_dim=16,
        num_heads=2,
        dropout=0.0,
    )
    out = block.forward_student(torch.randn(2, 5, 8))

    assert out["x_next"].shape == (2, 5, 8)
    assert out["x_post_attn"].shape == (2, 5, 8)
    assert out["z"].shape == (2, 5, 4)
    assert out["delta"].shape == (2, 5, 4)


def test_model_forward_returns_expected_shapes():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs["logits"].shape == (2, 4, 32)
    assert outputs["final_states"].shape == (2, 4, 8)
    assert outputs["loss"].ndim == 0
    assert outputs["loss_main"].ndim == 0
    assert outputs["loss_jepa"].ndim == 0
    assert len(outputs["z"]) == 3
    assert len(outputs["deltas"]) == 3
    assert len(outputs["targets"]) == 2


def test_model_forward_without_labels_uses_jepa_loss():
    model = IntertwinedHJEPA(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    outputs = model(input_ids=input_ids)

    assert outputs["loss_main"] is None
    assert torch.equal(outputs["loss"], make_config().lambda_jepa * outputs["loss_jepa"])


def test_depth_must_allow_future_layer_target():
    config = make_config()
    config.depth = 1

    try:
        IntertwinedHJEPA(config)
    except AssertionError as exc:
        assert "depth" in str(exc)
    else:
        raise AssertionError("Expected depth < 2 to be rejected")
