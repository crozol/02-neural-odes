"""Training loop para la Neural ODE."""

from __future__ import annotations

import torch
from torch import nn, optim

from .ode_net import ODEFunc, integrate


def train(
    y_train: torch.Tensor,
    t_train: torch.Tensor,
    dim: int = 2,
    hidden: int = 64,
    depth: int = 3,
    epochs: int = 500,
    lr: float = 1e-3,
    device: str = "cpu",
) -> ODEFunc:
    func = ODEFunc(dim=dim, hidden=hidden, depth=depth).to(device)
    opt = optim.Adam(func.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    y0 = y_train[0].to(device)
    t = t_train.to(device)
    y_true = y_train.to(device)

    for epoch in range(epochs):
        opt.zero_grad()
        y_pred = integrate(func, y0, t)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            print(f"[epoch {epoch:4d}] loss = {loss.item():.6f}")

    return func


if __name__ == "__main__":
    # TODO: generar trayectoria, entrenar, guardar modelo y gráfico
    raise NotImplementedError
