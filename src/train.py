"""Training loop and evaluation utilities for the Neural ODE.

The training objective is the mean squared error between the ground-truth
trajectory and the trajectory obtained by integrating the parametrised vector
field from the same initial condition over the same time grid.

A simple but effective trick is to stage the training horizon: we start by
matching only the first few time samples, and grow the window every
``grow_every`` epochs until the full training trajectory is included. This
avoids the well-known issue where the gradient through long integrations
vanishes or explodes early in training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, optim

from .ode_net import ODEFunc, count_parameters, integrate


# --------------------------------------------------------------------------- #
#  training
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    dim: int = 2
    hidden: int = 64
    depth: int = 3
    epochs: int = 1500
    lr: float = 5e-3
    weight_decay: float = 0.0
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7
    grow_every: int = 50          # epochs between curriculum window steps
    grow_start: int = 10          # initial window size in samples
    log_every: int = 50
    seed: int = 0
    device: str = "cpu"


@dataclass
class TrainResult:
    func: nn.Module
    loss_history: list[float] = field(default_factory=list)
    final_loss: float = float("nan")
    n_params: int = 0
    elapsed_s: float = 0.0
    config: TrainConfig | None = None


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(y_train: torch.Tensor, t_train: torch.Tensor,
          config: TrainConfig | None = None) -> TrainResult:
    """Fit a Neural ODE to ``(t_train, y_train)``.

    ``y_train`` must have shape ``(T, dim)``; ``t_train`` shape ``(T,)``.
    The initial condition is taken to be ``y_train[0]``.
    """
    if config is None:
        config = TrainConfig()
    _set_seed(config.seed)

    device = torch.device(config.device)
    y_train = y_train.to(device)
    t_train = t_train.to(device)

    func = ODEFunc(dim=config.dim, hidden=config.hidden,
                   depth=config.depth).to(device)
    opt = optim.Adam(func.parameters(), lr=config.lr,
                     weight_decay=config.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)
    loss_fn = nn.MSELoss()

    n_total = y_train.shape[0]
    window = min(max(config.grow_start, 2), n_total)

    history: list[float] = []
    t0 = time.time()

    for epoch in range(config.epochs):
        # curriculum: enlarge the matching window every ``grow_every`` epochs.
        if epoch > 0 and epoch % config.grow_every == 0 and window < n_total:
            window = min(window + max(config.grow_start, 2), n_total)

        opt.zero_grad()
        y0 = y_train[0]
        y_pred = integrate(func, y0, t_train[:window],
                           method=config.method,
                           rtol=config.rtol, atol=config.atol)
        loss = loss_fn(y_pred, y_train[:window])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=2.0)
        opt.step()
        sched.step()

        history.append(float(loss.detach().cpu()))
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            print(f"[epoch {epoch:5d} | win {window:4d}/{n_total}] "
                  f"loss = {history[-1]:.6e}   "
                  f"lr = {sched.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0
    return TrainResult(
        func=func.cpu(),
        loss_history=history,
        final_loss=history[-1],
        n_params=count_parameters(func),
        elapsed_s=elapsed,
        config=config,
    )


# --------------------------------------------------------------------------- #
#  evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EvalResult:
    t: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    mse_train: float
    mse_extrap: float
    rel_err_extrap: float


def evaluate(func: nn.Module, y0: np.ndarray, t_full: np.ndarray,
             y_true_full: np.ndarray, t_train_max: float) -> EvalResult:
    """Compare the learned dynamics against the ground truth on ``t_full``.

    Splits the loss into the training horizon (``t <= t_train_max``) and the
    extrapolation horizon (``t > t_train_max``) so we can quantify how well the
    network generalises beyond the data it was fit on.
    """
    func.eval()
    with torch.no_grad():
        y_pred = integrate(
            func,
            torch.tensor(y0, dtype=torch.float32),
            torch.tensor(t_full, dtype=torch.float32),
        ).cpu().numpy()

    train_mask = t_full <= t_train_max
    extrap_mask = ~train_mask

    mse_train = float(np.mean((y_pred[train_mask] - y_true_full[train_mask]) ** 2))
    mse_extrap = float(np.mean((y_pred[extrap_mask] - y_true_full[extrap_mask]) ** 2))

    denom = float(np.mean(y_true_full[extrap_mask] ** 2)) + 1e-12
    rel_err_extrap = float(np.sqrt(mse_extrap / denom))

    return EvalResult(
        t=t_full, y_true=y_true_full, y_pred=y_pred,
        mse_train=mse_train,
        mse_extrap=mse_extrap,
        rel_err_extrap=rel_err_extrap,
    )
