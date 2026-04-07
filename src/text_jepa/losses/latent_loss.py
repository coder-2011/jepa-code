import torch


def gather_target_states(target_states, target_positions):
    if target_states.ndim != 3:
        raise ValueError("target_states must have shape (B, L, D)")
    if target_positions.ndim != 2:
        raise ValueError("target_positions must have shape (B, T_max)")

    batch_size, sequence_length, hidden_dim = target_states.shape
    if target_positions.shape[0] != batch_size:
        raise ValueError("target_positions batch dimension must match target_states")
    if torch.any(target_positions < 0) or torch.any(target_positions >= sequence_length):
        raise ValueError("target_positions must be within [0, sequence_length)")

    # Expand positions across D so gather returns one latent vector per requested target slot.
    gather_index = target_positions.unsqueeze(-1).expand(-1, -1, hidden_dim)
    return torch.gather(target_states, dim=1, index=gather_index)


def masked_latent_mse(predictions, targets, target_valid_mask):
    if predictions.ndim != 3:
        raise ValueError("predictions must have shape (B, T_max, D)")
    if targets.ndim != 3:
        raise ValueError("targets must have shape (B, T_max, D)")
    if target_valid_mask.ndim != 2:
        raise ValueError("target_valid_mask must have shape (B, T_max)")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.shape[:2] != target_valid_mask.shape:
        raise ValueError("target_valid_mask must match the first two dimensions of predictions")

    valid_mask = target_valid_mask.to(torch.bool)
    if not torch.any(valid_mask):
        raise ValueError("target_valid_mask must include at least one valid target")

    # Squared error is computed densely first, then padded slots are zeroed out.
    squared_error = (predictions - targets).pow(2)
    expanded_mask = valid_mask.unsqueeze(-1).to(squared_error.dtype)
    masked_squared_error = squared_error * expanded_mask
    # Average over valid target vectors, not over padded slots.
    return masked_squared_error.sum() / expanded_mask.sum().mul(predictions.shape[-1])
