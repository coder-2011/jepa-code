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

    def forward(
        self,
        proj: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        per_layer: bool = False,
    ) -> torch.Tensor:
        assert proj.ndim >= 2, "SIGReg input must have shape (..., N, D)"
        assert proj.shape[-2] > 0 and proj.shape[-1] > 0, "SIGReg input must contain samples and features"
        assert not per_layer or proj.ndim >= 3, "per_layer SIGReg requires a leading layer dimension"

        proj = proj.float()
        if sample_mask is not None:
            assert sample_mask.shape == proj.shape[:-1], "sample_mask must match SIGReg input without the feature axis"
            sample_mask = sample_mask.to(device=proj.device, dtype=proj.dtype)

        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        with torch.no_grad():
            if per_layer:
                slices = torch.randn(proj.size(0), proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype)
                slices.div_(slices.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
            else:
                slices = torch.randn(proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype)
                slices.div_(slices.norm(p=2, dim=0).clamp_min(1e-12))

        projected = torch.einsum("l...d,lds->l...s", proj, slices) if per_layer else proj @ slices
        x_t = projected.unsqueeze(-1) * t
        if sample_mask is None:
            err = (x_t.cos().mean(dim=-3) - phi).square() + x_t.sin().mean(dim=-3).square()
            statistic = (err @ weights) * proj.size(-2)
            return statistic.flatten(1).mean(dim=1) if per_layer else statistic.mean()

        count = sample_mask.sum(dim=-1).clamp_min(1.0)
        mask = sample_mask.unsqueeze(-1).unsqueeze(-1)
        cos_mean = (x_t.cos() * mask).sum(dim=-3) / count.unsqueeze(-1).unsqueeze(-1)
        sin_mean = (x_t.sin() * mask).sum(dim=-3) / count.unsqueeze(-1).unsqueeze(-1)
        err = (cos_mean - phi).square() + sin_mean.square()
        statistic = (err @ weights) * count.unsqueeze(-1)
        valid_groups = sample_mask.any(dim=-1)
        assert valid_groups.any(), "sample_mask selects no SIGReg samples"
        if per_layer:
            denominator = valid_groups.flatten(1).sum(dim=1).clamp_min(1).to(statistic.dtype) * statistic.shape[-1]
            return (statistic * valid_groups.unsqueeze(-1)).flatten(1).sum(dim=1) / denominator
        return statistic[valid_groups].mean()
