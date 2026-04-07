import torch

from text_jepa.losses.latent_loss import gather_target_states, masked_latent_mse


def test_gather_target_states_returns_btd_shape():
    target_states = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    target_positions = torch.tensor([[1, 3], [0, 4]], dtype=torch.long)

    gathered = gather_target_states(target_states, target_positions)

    assert gathered.shape == (2, 2, 3)


def test_gather_target_states_matches_requested_positions():
    target_states = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
        ]
    )
    target_positions = torch.tensor([[2, 0], [1, 2]], dtype=torch.long)

    gathered = gather_target_states(target_states, target_positions)

    expected = torch.tensor(
        [
            [[4.0, 5.0], [0.0, 1.0]],
            [[8.0, 9.0], [10.0, 11.0]],
        ]
    )
    assert torch.equal(gathered, expected)


def test_gather_target_states_rejects_out_of_range_positions():
    target_states = torch.randn(2, 5, 3)
    target_positions = torch.tensor([[1, 5], [0, 4]], dtype=torch.long)

    try:
        gather_target_states(target_states, target_positions)
    except ValueError as exc:
        assert "within [0, sequence_length)" in str(exc)
    else:
        raise AssertionError("Expected gather_target_states to reject out-of-range positions")


def test_masked_latent_mse_uses_only_valid_target_slots():
    predictions = torch.tensor(
        [
            [[1.0, 1.0], [10.0, 10.0]],
            [[2.0, 2.0], [3.0, 3.0]],
        ]
    )
    targets = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    target_valid_mask = torch.tensor([[True, False], [True, True]])

    loss = masked_latent_mse(predictions, targets, target_valid_mask)

    expected = torch.tensor((1.0 + 1.0 + 4.0) / 3.0)
    assert torch.isclose(loss, expected)


def test_masked_latent_mse_rejects_zero_valid_targets():
    predictions = torch.zeros(2, 3, 4)
    targets = torch.zeros(2, 3, 4)
    target_valid_mask = torch.zeros(2, 3, dtype=torch.bool)

    try:
        masked_latent_mse(predictions, targets, target_valid_mask)
    except ValueError as exc:
        assert "at least one valid target" in str(exc)
    else:
        raise AssertionError("Expected masked_latent_mse to reject empty supervision")
