from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sliced Epps-Pulley SIGReg objective."""

    def __init__(self, knots: int = 17, num_slices: int = 256, t_max: float = 3.0) -> None:
        super().__init__()
        assert knots >= 3 and knots % 2 == 1, "knots must be an odd integer >= 3"
        assert num_slices > 0, "num_slices must be positive"
        assert t_max > 0, "t_max must be positive"

        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-0.5 * t.square())

        self.num_slices = num_slices
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        assert proj.ndim >= 2, "SIGReg input must have shape (..., N, D)"
        assert proj.shape[-2] > 0 and proj.shape[-1] > 0, "SIGReg input must contain samples and features"

        proj = proj.float()
        with torch.no_grad():
            slices = torch.randn(proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype)
            slices.div_(slices.norm(p=2, dim=0).clamp_min(1e-12))

        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        x_t = (proj @ slices).unsqueeze(-1) * t
        err = (x_t.cos().mean(dim=-3) - phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()
