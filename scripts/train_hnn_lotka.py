"""Hamiltonian Neural Network ablation on Lotka-Volterra.

The Lotka-Volterra system is canonically Hamiltonian in log-coordinates
``(q, p) = (ln x, ln z)`` with

    H(q, p) = b·e^p + d·e^q − a·p − c·q.

A Hamiltonian Neural Network parametrises a scalar ``H_θ(q, p)`` and
integrates the symplectic gradient, so by construction every predicted
orbit preserves ``H_θ`` along time. The contrast against the unconstrained
MLP baseline trained earlier is exactly what this script measures.

Usage:
    python -m scripts.train_hnn_lotka

Outputs:
    data/lotka_hnn_arrays.npz       — trajectories + loss history
    data/metrics.json               — appended with the lotka_volterra_hnn entry
    figures/hnn_loss.png            — HNN training loss
    figures/hnn_phase.png           — phase plane: truth · MLP · HNN
    figures/hnn_invariant.png       — H along time for all three
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
    LotkaParams,
    generate_trajectory,
    lotka_invariant,
    lotka_volterra,
)


def main() -> None:
    # Match the seed/params used by main.py for the unconstrained Lotka run
    seed = 1                                # main.py uses seed + 1 for lotka
    torch.manual_seed(seed)
    np.random.seed(seed)

    params = LotkaParams()
    y0 = np.array([10.0, 5.0])
    t_train_max = 12.0
    t_full_max = 22.0
    n_train = 300
    n_full = 550
    noise = 0.03

    t_train = np.linspace(0.0, t_train_max, n_train)
    t_full = np.linspace(0.0, t_full_max, n_full)

    y_train_obs = generate_trajectory(lotka_volterra, y0, t_train, params=params,
                                      noise=noise, seed=seed)
    y_full = generate_trajectory(lotka_volterra, y0, t_full, params=params)

    # Canonical log-coordinates. Clip away any rare negative observations
    # introduced by additive Gaussian noise (with σ = 0.03 on x ∼ 10 the
    # probability is tiny but it would crash log).
    eps = 1e-3
    qp_train_obs = np.log(np.clip(y_train_obs, eps, None))
    qp0 = np.log(np.clip(y0, eps, None))

    # ----- training -----
    print("=" * 64)
    print("ABLATION · Hamiltonian Neural Network on Lotka-Volterra")
    print("=" * 64)

    func = HNN(dim=2, hidden=96, depth=3)
    n_params = count_parameters(func)
    epochs = 2000
    lr_init = 2e-3
    grow_every = 80
    grow_start = 30
    log_every = 100
    method = "dopri5"
    rtol, atol = 1e-5, 1e-7

    opt = optim.Adam(func.parameters(), lr=lr_init)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    qp_train_t = torch.tensor(qp_train_obs, dtype=torch.float32)
    t_train_t = torch.tensor(t_train, dtype=torch.float32)

    history: list[float] = []
    n_total = qp_train_t.shape[0]
    window = min(max(grow_start, 2), n_total)

    t0 = time.time()
    for epoch in range(epochs):
        if epoch > 0 and epoch % grow_every == 0 and window < n_total:
            window = min(window + max(grow_start, 2), n_total)

        opt.zero_grad()
        qp_pred = integrate(func, qp_train_t[0], t_train_t[:window],
                            method=method, rtol=rtol, atol=atol)
        loss = loss_fn(qp_pred, qp_train_t[:window])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=2.0)
        opt.step()
        sched.step()

        history.append(float(loss.detach().cpu()))
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[epoch {epoch:5d} | win {window:4d}/{n_total}] "
                  f"loss = {history[-1]:.6e}   lr = {sched.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0

    # ----- evaluation -----
    # Note: HNN.forward needs autograd internally, so we don't wrap the
    # integration in torch.no_grad(). We just detach at the end.
    qp_pred_full_t = integrate(
        func, torch.tensor(qp0, dtype=torch.float32),
        torch.tensor(t_full, dtype=torch.float32),
        method=method, rtol=rtol, atol=atol,
    )
    qp_pred_full = qp_pred_full_t.detach().cpu().numpy()
    y_pred = np.exp(qp_pred_full)              # back to (x, z) coordinates

    train_mask = t_full <= t_train_max
    extrap_mask = ~train_mask
    mse_train = float(np.mean((y_pred[train_mask] - y_full[train_mask]) ** 2))
    mse_extrap = float(np.mean((y_pred[extrap_mask] - y_full[extrap_mask]) ** 2))
    scale = float(np.max(np.abs(y_full))) + 1e-12
    rel_err_extrap = float(np.sqrt(mse_extrap) / scale)

    invariant_true = lotka_invariant(y_full, params)
    invariant_pred_hnn = lotka_invariant(np.clip(y_pred, eps, None), params)
    invariant_drift = float(
        np.std(invariant_pred_hnn) / (np.abs(invariant_true.mean()) + 1e-12)
    )

    print(f"[done] HNN  -  train MSE={mse_train:.3e}  "
          f"extrap MSE={mse_extrap:.3e}  rel-err={rel_err_extrap:.3%}  "
          f"H drift={invariant_drift:.3%}")

    # ----- save NPZ -----
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        data_dir / "lotka_hnn_arrays.npz",
        t=t_full.astype(np.float32),
        y_true=y_full.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        invariant_true=invariant_true.astype(np.float32),
        invariant_pred=invariant_pred_hnn.astype(np.float32),
        loss_history=np.asarray(history, dtype=np.float32),
        t_train_max=np.float32(t_train_max),
        t_full_max=np.float32(t_full_max),
        grow_every=np.int32(grow_every),
        noise_obs=np.float32(noise),
    )

    # ----- update metrics.json (append or replace existing HNN entry) -----
    metrics_path = data_dir / "metrics.json"
    with open(metrics_path) as f:
        bundle = json.load(f)

    new_metric = {
        "system": "lotka_volterra_hnn",
        "model": "Hamiltonian Neural Network",
        "n_params": n_params,
        "epochs": epochs,
        "elapsed_s": round(elapsed, 2),
        "final_train_loss": history[-1],
        "mse_train": mse_train,
        "mse_extrap": mse_extrap,
        "rel_err_extrap": rel_err_extrap,
        "invariant_drift_relative": invariant_drift,
        "t_train_max": t_train_max,
        "t_full_max": t_full_max,
        "noise_obs": noise,
    }
    bundle["experiments"] = [
        m for m in bundle["experiments"]
        if m.get("system") != "lotka_volterra_hnn"
    ]
    bundle["experiments"].append(new_metric)
    with open(metrics_path, "w") as f:
        json.dump(bundle, f, indent=2)

    # ----- comparison figures -----
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # MLP results from the previous main.py run
    mlp = np.load(data_dir / "lotka_arrays.npz")

    plots.plot_loss(
        history, str(fig_dir / "hnn_loss.png"),
        system_name="HNN · Lotka-Volterra",
        grow_every=grow_every,
    )
    plots.plot_phase_triple(
        y_true=y_full,
        y_mlp=mlp["y_pred"],
        y_hnn=y_pred,
        vector_field=(mlp["vf_X"], mlp["vf_Y"], mlp["vf_U"], mlp["vf_V"]),
        out_path=str(fig_dir / "hnn_phase.png"),
    )
    plots.plot_invariant_comparison(
        t=t_full,
        H_true=invariant_true,
        H_mlp=mlp["invariant_pred"],
        H_hnn=invariant_pred_hnn,
        t_train_max=t_train_max,
        out_path=str(fig_dir / "hnn_invariant.png"),
    )

    print("[ok] HNN ablation complete.")


if __name__ == "__main__":
    main()
