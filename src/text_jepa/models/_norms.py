from torch import nn


def make_rms_norm(hidden_dim):
    norm = nn.RMSNorm(hidden_dim)
    norm.bias = None
    return norm
