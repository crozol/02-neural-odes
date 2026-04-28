"""End-to-end pipeline: generate synthetic data, train the Neural ODE,
evaluate extrapolation, and write all figures + a JSON metrics summary.

Two systems are run back-to-back:

    1. damped pendulum      (default parameters in src.systems.PendulumParams)
    2. Lotka-Volterra       (default parameters in src.systems.LotkaParams)

Outputs go to ``figures/`` (the PNGs embedded in the README) and ``data/`` (the
JSON metrics summary). Both directories are created on demand.

Usage:
    python main.py
    python main.py --system pendulum
    python main.py --system lotka
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import plots
from src.ode_net import vector_field_grid
from src.systems import (
    LotkaParams,
    PendulumParams,
    damped_pendulum,
    generate_trajectory,
    lotka_invariant,
    lotka_volterra,
    pendulum_energy,
)
from src.train import TrainConfig, evaluate, train


# --------------------------------------------------------------------------- #
#  experiment definitions
# --------------------------------------------------------------------------- #

def run_pendulum(out_dir: Path, data_dir: Path, seed: int = 0) -> dict:
    print("\n" + "=" * 64)
    print("EXPERIMENT 1/2 · damped pendulum")
    print("=" * 64)

    params = PendulumParams()
    y0 = np.array([2.0, 0.0])              # large initial angle, no angular velocity
    t_train_max = 12.0
    t_full_max = 20.0
    n_train = 360
    n_full = 600
    noise = 0.005

    t_train = np.linspace(0.0, t_train_max, n_train)
    t_full = np.linspace(0.0, t_full_max, n_full)

    y_train_obs = generate_trajectory(damped_pendulum, y0, t_train, params=params,
                                      noise=noise, seed=seed)
    y_full = generate_trajectory(damped_pendulum, y0, t_full, params=params)

    cfg = TrainConfig(
        dim=2, hidden=96, depth=3,
        epochs=2500, lr=4e-3,
        grow_every=70, grow_start=20,
        log_every=100, seed=seed,
    )
    res = train(
        torch.tensor(y_train_obs, dtype=torch.float32),
        torch.tensor(t_train, dtype=torch.float32),
        cfg,
    )

    ev = evaluate(res.func, y0, t_full, y_full, t_train_max=t_train_max)

    # learned vector field on the orbit's bounding box (with a margin)
    pad = 0.6
    x_lo, x_hi = float(y_full[:, 0].min() - pad), float(y_full[:, 0].max() + pad)
    y_lo, y_hi = float(y_full[:, 1].min() - pad), float(y_full[:, 1].max() + pad)
    vf = vector_field_grid(res.func, (x_lo, x_hi), (y_lo, y_hi), n=24)

    energy_true = pendulum_energy(ev.y_true, params)
    energy_pred = pendulum_energy(ev.y_pred, params)

    plots.plot_loss(res.loss_history, str(out_dir / "pendulum_loss.png"),
                    system_name="damped pendulum",
                    grow_every=cfg.grow_every)
    plots.plot_time_series(ev.t, ev.y_true, ev.y_pred, t_train_max,
                           str(out_dir / "pendulum_traj.png"))
    plots.plot_phase(ev.y_true, ev.y_pred, vf,
                     str(out_dir / "pendulum_phase.png"))
    plots.plot_energy(ev.t, energy_true, energy_pred, t_train_max,
                      str(out_dir / "pendulum_energy.png"))

    energy_drift = float(abs(energy_pred[-1] - energy_true[-1]) /
                         (abs(energy_true[-1]) + 1e-12))

    metrics = dict(
        system="damped_pendulum",
        n_params=res.n_params,
        epochs=cfg.epochs,
        elapsed_s=round(res.elapsed_s, 2),
        final_train_loss=res.final_loss,
        mse_train=ev.mse_train,
        mse_extrap=ev.mse_extrap,
        rel_err_extrap=ev.rel_err_extrap,
        energy_drift_final=energy_drift,
        t_train_max=t_train_max,
        t_full_max=t_full_max,
        noise_obs=noise,
    )
    print(f"[done] pendulum  -  train MSE={ev.mse_train:.3e}  "
          f"extrap MSE={ev.mse_extrap:.3e}  rel-err={ev.rel_err_extrap:.3%}  "
          f"|dE/E|={energy_drift:.2%}")
    return metrics


def run_lotka(out_dir: Path, data_dir: Path, seed: int = 0) -> dict:
    print("\n" + "=" * 64)
    print("EXPERIMENT 2/2 · Lotka-Volterra")
    print("=" * 64)

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

    cfg = TrainConfig(
        dim=2, hidden=96, depth=3,
        epochs=2000, lr=2e-3,
        grow_every=80, grow_start=30,
        log_every=100, seed=seed + 1,
    )
    res = train(
        torch.tensor(y_train_obs, dtype=torch.float32),
        torch.tensor(t_train, dtype=torch.float32),
        cfg,
    )

    ev = evaluate(res.func, y0, t_full, y_full, t_train_max=t_train_max)

    pad = 1.5
    x_lo, x_hi = float(max(0.05, y_full[:, 0].min() - pad)), float(y_full[:, 0].max() + pad)
    y_lo, y_hi = float(max(0.05, y_full[:, 1].min() - pad)), float(y_full[:, 1].max() + pad)
    vf = vector_field_grid(res.func, (x_lo, x_hi), (y_lo, y_hi), n=24)

    invariant_true = lotka_invariant(ev.y_true, params)
    invariant_pred = lotka_invariant(np.clip(ev.y_pred, 1e-3, None), params)

    plots.plot_loss(res.loss_history, str(out_dir / "lotka_loss.png"),
                    system_name="Lotka-Volterra",
                    grow_every=cfg.grow_every)
    plots.plot_lotka_time(ev.t, ev.y_true, ev.y_pred, t_train_max,
                          str(out_dir / "lotka_traj.png"))
    plots.plot_lotka_phase(ev.y_true, ev.y_pred, vf,
                           str(out_dir / "lotka_phase.png"))

    invariant_drift = float(np.std(invariant_pred) / (np.abs(invariant_true.mean()) + 1e-12))

    metrics = dict(
        system="lotka_volterra",
        n_params=res.n_params,
        epochs=cfg.epochs,
        elapsed_s=round(res.elapsed_s, 2),
        final_train_loss=res.final_loss,
        mse_train=ev.mse_train,
        mse_extrap=ev.mse_extrap,
        rel_err_extrap=ev.rel_err_extrap,
        invariant_drift_relative=invariant_drift,
        t_train_max=t_train_max,
        t_full_max=t_full_max,
        noise_obs=noise,
    )
    print(f"[done] lotka  -  train MSE={ev.mse_train:.3e}  "
          f"extrap MSE={ev.mse_extrap:.3e}  rel-err={ev.rel_err_extrap:.3%}")
    return metrics


# --------------------------------------------------------------------------- #
#  entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["pendulum", "lotka", "both"],
                        default="both", help="experiment to run (default: both)")
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   experiments=[])

    if args.system in ("pendulum", "both"):
        summary["experiments"].append(run_pendulum(out_dir, data_dir, args.seed))
    if args.system in ("lotka", "both"):
        summary["experiments"].append(run_lotka(out_dir, data_dir, args.seed))

    summary_path = data_dir / "metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ok] {summary_path}")
    print(f"[ok] figures in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
