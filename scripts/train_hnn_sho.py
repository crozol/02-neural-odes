"""Hamiltonian Neural Network sanity check on the simple harmonic oscillator.

The undamped simple harmonic oscillator is the trivial limit at which an HNN
must succeed: the state ``(q, p)`` is already canonical and the true
Hamiltonian ``H(q, p) = ½(p² + ω²q²)`` is a quadratic form, so an MLP of
modest width can approximate it cleanly from a single short noisy trajectory.
Once trained, integrating the symplectic gradient of the learned ``H_θ`` for
many periods produces an extrapolation whose drift in ``H`` is bounded by the
truncation error of the non-symplectic ``dopri5`` solver — sub-1% over four
training periods is the expected order of magnitude with the hyperparameters
below.

This sits at the opposite end of the rigour spectrum from the Lotka-Volterra
ablation: Lotka is the realistic case (log-coordinate transform, nonlinear
canonical Hamiltonian, finite-capacity drift); SHO is the textbook
sanity-check that the architecture is implemented correctly.

Usage:
    python -m scripts.train_hnn_sho

Outputs:
    data/sho_hnn_arrays.npz         — trajectories + loss history + H values
    data/metrics.json               — appended with the sho_hnn entry
    figures/sho_invariant.png       — H(t) for the true vs HNN-predicted orbit
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from src import plots
from src.hnn import HNN, count_parameters
from src.ode_net import integrate
from src.systems import (
    SHOParams,
    generate_trajectory,
    simple_harmonic,
    sho_energy,
)


def main() -> None:
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    params = SHOParams(omega=1.0)
    y0 = np.array([1.0, 0.0])               # unit amplitude at t = 0, zero momentum
    period = 2.0 * np.pi / params.omega     # ≈ 6.283
    t_train_max = period                    # one full period of training data
    t_full_max = 4.0 * period               # four full periods total — the extrapolation test
    n_train = 200
    n_full = 800
    noise = 0.005

    t_train = np.linspace(0.0, t_train_max, n_train)
    t_full = np.linspace(0.0, t_full_max, n_full)

    y_train_obs = generate_trajectory(simple_harmonic, y0, t_train, params=params,
                                      noise=noise, seed=seed)
    y_full = generate_trajectory(simple_harmonic, y0, t_full, params=params)

    print("=" * 64)
    print("SANITY CHECK · Hamiltonian Neural Network on simple harmonic oscillator")
    print("=" * 64)

    # Match the Lotka HNN ablation (hidden=96, depth=3) so the comparison
    # between Lotka HNN and SHO HNN drift is a property of the system, not of
    # network capacity. dopri5 is non-symplectic — the residual H drift below
    # is the solver-induced truncation floor for these hyperparameters.
    func = HNN(dim=2, hidden=96, depth=3)
    n_params = count_parameters(func)
    epochs = 1500
    lr_init = 2e-3
    grow_every = 60
    grow_start = 30
    log_every = 100
    method = "dopri5"
    rtol, atol = 1e-7, 1e-9

    opt = optim.Adam(func.parameters(), lr=lr_init)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    y_train_t = torch.tensor(y_train_obs, dtype=torch.float32)
    t_train_t = torch.tensor(t_train, dtype=torch.float32)

    history: list[float] = []
    n_total = y_train_t.shape[0]
    window = min(max(grow_start, 2), n_total)

    t0 = time.time()
    for epoch in range(epochs):
        if epoch > 0 and epoch % grow_every == 0 and window < n_total:
            window = min(window + max(grow_start, 2), n_total)

        opt.zero_grad()
        y_pred_train = integrate(func, y_train_t[0], t_train_t[:window],
                                 method=method, rtol=rtol, atol=atol)
        loss = loss_fn(y_pred_train, y_train_t[:window])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=2.0)
        opt.step()
        sched.step()

        history.append(float(loss.detach().cpu()))
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[epoch {epoch:4d} | win {window:3d}/{n_total}] "
                  f"loss = {history[-1]:.6e}   lr = {sched.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0

    # ----- evaluation over the full extrapolation horizon -----
    y_pred_full_t = integrate(
        func, torch.tensor(y0, dtype=torch.float32),
        torch.tensor(t_full, dtype=torch.float32),
        method=method, rtol=rtol, atol=atol,
    )
    y_pred = y_pred_full_t.detach().cpu().numpy()

    train_mask = t_full <= t_train_max
    extrap_mask = ~train_mask
    mse_train = float(np.mean((y_pred[train_mask] - y_full[train_mask]) ** 2))
    mse_extrap = float(np.mean((y_pred[extrap_mask] - y_full[extrap_mask]) ** 2))
    scale = float(np.max(np.abs(y_full))) + 1e-12
    rel_err_extrap = float(np.sqrt(mse_extrap) / scale)

    H_true = sho_energy(y_full, params)
    H_pred = sho_energy(y_pred, params)
    H_drift_relative = float(
        np.std(H_pred) / (np.abs(H_true.mean()) + 1e-12)
    )
    H_drift_max_abs = float(np.max(np.abs(H_pred - H_true.mean())))

    print(f"[done] HNN/SHO  -  train MSE={mse_train:.3e}  "
          f"extrap MSE={mse_extrap:.3e}  rel-err={rel_err_extrap:.3%}  "
          f"H drift (std/mean) = {H_drift_relative:.3%}  "
          f"H max abs error = {H_drift_max_abs:.3e}")

    # ----- save NPZ -----
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        data_dir / "sho_hnn_arrays.npz",
        t=t_full.astype(np.float32),
        y_true=y_full.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        H_true=H_true.astype(np.float32),
        H_pred=H_pred.astype(np.float32),
        loss_history=np.asarray(history, dtype=np.float32),
        t_train_max=np.float32(t_train_max),
        t_full_max=np.float32(t_full_max),
        omega=np.float32(params.omega),
        noise_obs=np.float32(noise),
    )

    # ----- update metrics.json (append or replace existing SHO entry) -----
    metrics_path = data_dir / "metrics.json"
    with open(metrics_path) as f:
        bundle = json.load(f)

    new_metric = {
        "system": "simple_harmonic_hnn",
        "model": "Hamiltonian Neural Network",
        "n_params": n_params,
        "epochs": epochs,
        "elapsed_s": round(elapsed, 2),
        "final_train_loss": history[-1],
        "mse_train": mse_train,
        "mse_extrap": mse_extrap,
        "rel_err_extrap": rel_err_extrap,
        "invariant_drift_relative": H_drift_relative,
        "invariant_max_abs_error": H_drift_max_abs,
        "t_train_max": float(t_train_max),
        "t_full_max": float(t_full_max),
        "noise_obs": noise,
    }
    bundle["experiments"] = [
        m for m in bundle["experiments"]
        if m.get("system") != "simple_harmonic_hnn"
    ]
    bundle["experiments"].append(new_metric)
    with open(metrics_path, "w") as f:
        json.dump(bundle, f, indent=2)

    # ----- invariant comparison figure -----
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    plots.plot_energy(
        t=t_full,
        energy_true=H_true,
        energy_pred=H_pred,
        t_train_max=float(t_train_max),
        out_path=str(fig_dir / "sho_invariant.png"),
    )

    print("[ok] SHO HNN sanity check complete.")


if __name__ == "__main__":
    main()
