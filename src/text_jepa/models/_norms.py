from torch import nn


def make_rms_norm(hidden_dim):
    # PyTorch's RMSNorm exposes a bias attribute even though the JEPA stack keeps normalization affine-light.
    norm = nn.RMSNorm(hidden_dim)
    norm.bias = None
    return norm
