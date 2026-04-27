"""Neural network parametrising the vector field of an ODE.

The model is a small MLP ``f_theta : R^d x R -> R^d`` with smooth activations
(``tanh``), trained so that the trajectories obtained by integrating
``dy/dt = f_theta(y, t)`` match the observed ones.

Time enters the network only as an extra feature so the architecture itself is
autonomous by default — for the systems in this project (pendulum,
Lotka-Volterra) the true vector field has no explicit time dependence either.
The integration uses ``torchdiffeq.odeint`` (Dormand-Prince adaptive RK45 by
default), which is differentiable end-to-end via the adjoint method or direct
backprop through the solver steps.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """MLP approximating the right-hand side of an autonomous ODE.

    Parameters
    ----------
    dim:
        Dimension of the state vector ``y``.
    hidden:
        Width of every hidden layer.
    depth:
        Number of *hidden* layers (so the total Linear count is ``depth + 1``).
    activation:
        Activation factory; defaults to ``nn.Tanh`` (smooth, bounded — works
        well with adaptive ODE solvers).
    """

    def __init__(self, dim: int = 2, hidden: int = 64, depth: int = 3,
                 activation: type[nn.Module] = nn.Tanh):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = [nn.Linear(dim, hidden), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

        # Small final-layer init keeps the initial vector field close to zero,
        # which empirically stabilises the first few training epochs.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.net(y)


# --------------------------------------------------------------------------- #
#  integration helpers
# --------------------------------------------------------------------------- #

def integrate(func: nn.Module, y0: torch.Tensor, t: torch.Tensor, *,
              method: str = "dopri5", rtol: float = 1e-5,
              atol: float = 1e-7) -> torch.Tensor:
    """Integrate ``dy/dt = func(t, y)`` from ``y0`` over time grid ``t``."""
    return odeint(func, y0, t, method=method, rtol=rtol, atol=atol)


def vector_field_grid(func: nn.Module, x_range: Sequence[float],
                      y_range: Sequence[float], n: int = 24,
                      device: str = "cpu") -> tuple[torch.Tensor, ...]:
    """Sample ``func(0, y)`` on a regular grid for plotting purposes.

    Returns ``(X, Y, U, V)`` ready to feed ``matplotlib.pyplot.streamplot``.
    """
    xs = torch.linspace(x_range[0], x_range[1], n, device=device)
    ys = torch.linspace(y_range[0], y_range[1], n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    with torch.no_grad():
        f = func(torch.zeros(()), grid)
    U = f[:, 0].reshape(n, n)
    V = f[:, 1].reshape(n, n)
    return X, Y, U, V


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
