"""Red neuronal que aproxima el campo vectorial dy/dt = f_theta(y, t)."""

from __future__ import annotations

import torch
from torch import nn


class ODEFunc(nn.Module):
    def __init__(self, dim: int = 2, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


def integrate(func: ODEFunc, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Integra la EDO dy/dt = func(t, y) usando torchdiffeq."""
    from torchdiffeq import odeint

    return odeint(func, y0, t, method="dopri5", rtol=1e-5, atol=1e-7)
