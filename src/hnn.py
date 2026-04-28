"""Hamiltonian Neural Network for 2D autonomous systems.

The network parametrises a scalar Hamiltonian ``H_θ(q, p)`` and the dynamics
are computed via autograd as the symplectic gradient

    dq/dt = +∂H/∂p
    dp/dt = -∂H/∂q

By construction every trajectory produced by integrating this field
preserves ``H_θ`` (up to the truncation error of the solver — for symplectic
integrators it is exact, for ``dopri5`` over the horizons used here it is
much smaller than the drift of an unconstrained MLP).

For the Lotka-Volterra system, canonical coordinates are
``(q, p) = (ln x, ln z)`` and the true Hamiltonian is

    H(q, p) = b·e^p + d·e^q − a·p − c·q

so an HNN with enough capacity should recover the orbit's invariant up to
the noise floor — the contrast against the unconstrained MLP baseline is
what this class is built to quantify.
"""

from __future__ import annotations

import torch
from torch import nn


class HNN(nn.Module):
    """Hamiltonian Neural Network on a 2D phase space.

    Parameters
    ----------
    dim:
        Dimension of the state vector. Must be 2 (one ``q`` and one ``p``).
    hidden:
        Width of every hidden layer.
    depth:
        Number of hidden Linear layers (so the total Linear count is
        ``depth + 1``).
    activation:
        Activation factory; defaults to ``nn.Tanh`` (smooth, twice
        differentiable — required because we backprop through ``∇H``).
    """

    def __init__(self, dim: int = 2, hidden: int = 96, depth: int = 3,
                 activation: type[nn.Module] = nn.Tanh):
        super().__init__()
        if dim != 2:
            raise NotImplementedError("HNN is implemented only for 2D systems.")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = [nn.Linear(dim, hidden), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation()]
        layers += [nn.Linear(hidden, 1)]
        self.H = nn.Sequential(*layers)

        # Same trick as the unconstrained MLP: small final-layer init keeps
        # the initial vector field close to zero everywhere, which prevents
        # the very first integration from blowing up.
        with torch.no_grad():
            self.H[-1].weight.mul_(0.1)
            self.H[-1].bias.zero_()

    def hamiltonian(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the scalar Hamiltonian at ``y`` (shape ``(..., 2)``)."""
        return self.H(y).squeeze(-1)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        # We need ``y.requires_grad`` to compute ∂H/∂y via autograd; if the
        # caller passed a detached tensor (typical at evaluation time), clone
        # and re-enable grad without contaminating the original.
        with torch.enable_grad():
            if not y.requires_grad:
                y = y.detach().requires_grad_(True)
            H_val = self.H(y).sum()
            grad_H, = torch.autograd.grad(
                H_val, y, create_graph=self.training,
            )
        # Symplectic gradient: dq/dt = +∂H/∂p, dp/dt = -∂H/∂q.
        return torch.stack([grad_H[..., 1], -grad_H[..., 0]], dim=-1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
