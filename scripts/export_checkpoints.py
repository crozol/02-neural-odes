"""Train and export the model checkpoints consumed by the HNN-demo Space.

Produces four ``.pt`` checkpoints under ``--out-dir`` — an unconstrained MLP
baseline and a Hamiltonian NN for each of the two conservative systems
(Lotka-Volterra and the simple harmonic oscillator). Each bundle carries the
architecture config, the trained weights, and the coordinate convention the
demo app needs to integrate the field and map it back to the native phase
space (``coords = "log"`` for the Lotka HNN, which lives in ``(ln x, ln z)``).

The configurations mirror the ones documented in the project README so the
checkpoints reproduce the reported invariant-drift numbers. Run from the
project root:

    python -m scripts.export_checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from src.hnn import HNN, count_parameters
from src.ode_net import ODEFunc, integrate
from src.systems import (
    LotkaParams,
    SHOParams,
    generate_trajectory,
    lotka_invariant,
    lotka_volterra,
    sho_energy,
    simple_harmonic,
)
from src.train import TrainConfig, train


def _train_hnn(y_train_obs, t_train, *, hidden, depth, epochs, lr_init,
               grow_every, grow_start, method, rtol, atol, seed):
    """Curriculum-trained HNN, mirroring scripts/train_hnn_*.py."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    func = HNN(dim=2, hidden=hidden, depth=depth)
    opt = optim.Adam(func.parameters(), lr=lr_init)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    y_t = torch.tensor(y_train_obs, dtype=torch.float32)
    t_t = torch.tensor(t_train, dtype=torch.float32)
    n_total = y_t.shape[0]
    window = min(max(grow_start, 2), n_total)

    for epoch in range(epochs):
        if epoch > 0 and epoch % grow_every == 0 and window < n_total:
            window = min(window + max(grow_start, 2), n_total)
        opt.zero_grad()
        pred = integrate(func, y_t[0], t_t[:window],
                         method=method, rtol=rtol, atol=atol)
        loss = loss_fn(pred, y_t[:window])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=2.0)
        opt.step()
        sched.step()
        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"  [hnn epoch {epoch:4d}/{epochs}] loss = {float(loss):.4e}")
    return func.cpu()


def _integrate_native(func, y0_model, t, *, coords, method="dopri5",
                      rtol=1e-5, atol=1e-7):
    """Integrate a model and return the trajectory in native (x,z)/(q,p) coords."""
    pred = integrate(func, torch.tensor(y0_model, dtype=torch.float32),
                     torch.tensor(t, dtype=torch.float32),
                     method=method, rtol=rtol, atol=atol).detach().cpu().numpy()
    return np.exp(pred) if coords == "log" else pred


def _save(path, *, model_type, system, func, hidden, depth, coords, y0_model):
    payload = {
        "model_type": model_type,
        "system": system,
        "config": {"dim": 2, "hidden": hidden, "depth": depth},
        "state_dict": func.state_dict(),
        "coords": coords,
        "y0_model": [float(v) for v in y0_model],
        "n_params": count_parameters(func),
    }
    torch.save(payload, path)
    print(f"[ok] saved {path}  ({count_parameters(func)} params, coords={coords})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="../spaces-neural-odes-hnn/checkpoints")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ===================== Lotka-Volterra =====================
    lp = LotkaParams()
    y0_lotka = np.array([10.0, 5.0])
    t_train_l = np.linspace(0.0, 12.0, 300)
    t_full_l = np.linspace(0.0, 22.0, 550)
    y_full_l = generate_trajectory(lotka_volterra, y0_lotka, t_full_l, params=lp)
    H_true_l = lotka_invariant(y_full_l, lp)

    print("=" * 64, "\nMLP baseline · Lotka-Volterra\n", "=" * 64, sep="")
    y_obs = generate_trajectory(lotka_volterra, y0_lotka, t_train_l,
                                params=lp, noise=0.03, seed=0)
    cfg = TrainConfig(dim=2, hidden=96, depth=3, epochs=2000, lr=2e-3,
                      grow_every=80, grow_start=30, log_every=200, seed=1)
    res = train(torch.tensor(y_obs, dtype=torch.float32),
                torch.tensor(t_train_l, dtype=torch.float32), cfg)
    y_pred = _integrate_native(res.func, y0_lotka, t_full_l, coords="direct")
    drift = float(np.std(lotka_invariant(np.clip(y_pred, 1e-3, None), lp))
                  / (abs(H_true_l.mean()) + 1e-12))
    print(f"  -> MLP Lotka H-drift = {drift:.2%}  (README ~7.9%)")
    _save(out / "mlp_lotka.pt", model_type="mlp", system="lotka", func=res.func,
          hidden=96, depth=3, coords="direct", y0_model=y0_lotka)

    print("=" * 64, "\nHNN · Lotka-Volterra (log-coords)\n", "=" * 64, sep="")
    y_obs_h = generate_trajectory(lotka_volterra, y0_lotka, t_train_l,
                                  params=lp, noise=0.03, seed=1)
    qp_obs = np.log(np.clip(y_obs_h, 1e-3, None))
    qp0 = np.log(np.clip(y0_lotka, 1e-3, None))
    func_hl = _train_hnn(qp_obs, t_train_l, hidden=96, depth=3, epochs=2000,
                         lr_init=2e-3, grow_every=80, grow_start=30,
                         method="dopri5", rtol=1e-5, atol=1e-7, seed=1)
    y_pred_h = _integrate_native(func_hl, qp0, t_full_l, coords="log")
    drift_h = float(np.std(lotka_invariant(np.clip(y_pred_h, 1e-3, None), lp))
                    / (abs(H_true_l.mean()) + 1e-12))
    print(f"  -> HNN Lotka H-drift = {drift_h:.2%}  (README ~2.4%)")
    _save(out / "hnn_lotka.pt", model_type="hnn", system="lotka", func=func_hl,
          hidden=96, depth=3, coords="log", y0_model=qp0)

    # ===================== Simple harmonic oscillator =====================
    sp = SHOParams(omega=1.0)
    y0_sho = np.array([1.0, 0.0])
    period = 2.0 * np.pi / sp.omega
    t_train_s = np.linspace(0.0, period, 200)
    t_full_s = np.linspace(0.0, 4.0 * period, 800)
    y_full_s = generate_trajectory(simple_harmonic, y0_sho, t_full_s, params=sp)
    H_true_s = sho_energy(y_full_s, sp)

    print("=" * 64, "\nMLP baseline · simple harmonic oscillator\n", "=" * 64, sep="")
    y_obs_s = generate_trajectory(simple_harmonic, y0_sho, t_train_s,
                                  params=sp, noise=0.005, seed=0)
    cfg_s = TrainConfig(dim=2, hidden=96, depth=3, epochs=1500, lr=2e-3,
                        grow_every=60, grow_start=30, log_every=200, seed=0)
    res_s = train(torch.tensor(y_obs_s, dtype=torch.float32),
                  torch.tensor(t_train_s, dtype=torch.float32), cfg_s)
    y_pred_s = _integrate_native(res_s.func, y0_sho, t_full_s, coords="direct")
    drift_s = float(np.std(sho_energy(y_pred_s, sp)) / (abs(H_true_s.mean()) + 1e-12))
    print(f"  -> MLP SHO H-drift = {drift_s:.2%}")
    _save(out / "mlp_sho.pt", model_type="mlp", system="sho", func=res_s.func,
          hidden=96, depth=3, coords="direct", y0_model=y0_sho)

    print("=" * 64, "\nHNN · simple harmonic oscillator\n", "=" * 64, sep="")
    func_hs = _train_hnn(y_obs_s, t_train_s, hidden=96, depth=3, epochs=1500,
                         lr_init=2e-3, grow_every=60, grow_start=30,
                         method="dopri5", rtol=1e-7, atol=1e-9, seed=0)
    y_pred_hs = _integrate_native(func_hs, y0_sho, t_full_s, coords="direct",
                                  rtol=1e-7, atol=1e-9)
    drift_hs = float(np.std(sho_energy(y_pred_hs, sp)) / (abs(H_true_s.mean()) + 1e-12))
    print(f"  -> HNN SHO H-drift = {drift_hs:.2%}  (README ~0.92%)")
    _save(out / "hnn_sho.pt", model_type="hnn", system="sho", func=func_hs,
          hidden=96, depth=3, coords="direct", y0_model=y0_sho)

    print("\n[done] all four checkpoints exported to", out.resolve())


if __name__ == "__main__":
    main()
