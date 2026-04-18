from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SlicedEppsPulleySIGReg(nn.Module):
    """
    Local copy of LeJEPA's sliced Epps-Pulley SIGReg objective.

    Args:
        num_slices: number of random unit 1D projections.
        t_max: upper integration bound over [0, t_max]; symmetry handles negative t.
        n_points: odd number of trapezoid integration knots.

    Input:
        z: (..., N, D) or any tensor with trailing feature width D. This module
           flattens all leading sample dimensions into N and returns a scalar.
    """

    def __init__(
        self,
        num_slices: int = 256,
        t_max: float = 3.0,
        n_points: int = 17,
    ) -> None:
        super().__init__()
        assert num_slices > 0, "num_slices must be positive"
        assert t_max > 0, "t_max must be positive"
        assert n_points >= 3 and n_points % 2 == 1, "n_points must be an odd integer >= 3"

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())

        self.num_slices = num_slices
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("weights", weights * phi, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.shape[-1] > 0, "SIGReg input must have a feature dimension"

        x = z.float().reshape(-1, z.shape[-1])
        assert x.shape[0] > 0, "SIGReg input must contain at least one sample"
        x = x * x.pow(2).mean(dim=-1, keepdim=True).add(1e-8).rsqrt()

        with torch.no_grad():
            slices = F.normalize(
                torch.randn((x.shape[-1], self.num_slices), device=x.device, dtype=x.dtype),
                p=2,
                dim=0,
            )

        projected = x @ slices
        t = self.t.to(device=x.device, dtype=x.dtype)
        phi = self.phi.to(device=x.device, dtype=x.dtype)
        weights = self.weights.to(device=x.device, dtype=x.dtype)
        x_t = projected.unsqueeze(-1) * t
        cos_mean = torch.cos(x_t).mean(dim=0)
        sin_mean = torch.sin(x_t).mean(dim=0)
        err = (cos_mean - phi).square() + sin_mean.square()
        statistic = (err @ weights) * x.shape[0]
        return statistic.mean()
