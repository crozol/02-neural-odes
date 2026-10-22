"""Lotka-Volterra ablation over models, training regimes and seeds, scored on
initial conditions the models never saw.

The first ablation (scripts/train_hnn_lotka.py) trained each model on one
trajectory and scored it on the forward extrapolation of that same orbit.
That cannot tell whether a model learned the vector field or only that orbit,
and it left three gaps this script closes:

* **Held-out initial conditions.** Every model is scored on five orbits that
  are not in its training data: two inside the range of the conserved
  quantity H covered by the multi-trajectory training set, one below it
  (closer to the fixed point) and two above it.
* **A coordinate control.** The HNN works in log-coordinates (q, p) =
  (ln x, ln z) while the baseline MLP works in (x, z), so the earlier
  comparison mixed two changes. ``mlp_log`` is the same MLP trained in
  (q, p) with the same loss as the HNN, which isolates the Hamiltonian
  structure from the change of coordinates.
* **Learned initial state.** Training integrates from the first *noisy*
  observation in the original scripts. Here the initial state of every
  training trajectory is a free parameter, initialised at that observation
  and fitted jointly with the network.
* **Prior on the initial state** (``--y0-prior``). A free y0 can drift far
  from its observation (up to 0.2 in (x, z), about 7 σ) to absorb model
  error. With the prior, the loss is the Gaussian MAP objective

      Σ_t |pred − obs|² / (2 s²) + Σ |y0 − y_obs(0)|² / (2 s²),

  divided by the number of residuals so that its scale matches the MSE used
  without the prior; s is the observation noise in model coordinates (σ in
  (x, z), σ / y_obs(0) in log-coordinates by the delta method). The prior
  counts the first observation a second time, so it is a weak constraint by
  construction: a large drift that survives it is driven by the data. Runs
  go to data/ablation_ic_prior/. Every run reports the displacement
  |y0 − y_obs(0)| / s per coordinate.

Models: ``mlp`` (MLP in (x, z), loss in (x, z)), ``mlp_log`` (MLP in (q, p),
loss in (q, p)), ``hnn`` (HNN in (q, p), loss in (q, p)); all ~19 000
parameters, 3 hidden layers of 96, tanh. Regimes: ``single`` (one trajectory
from (10, 5), as before) and ``multi`` (three trajectories, from (10, 5),
(6, 4) and (12, 6)). Every run: 2000 Adam epochs, cosine lr from 2e-3,
gradient clipping at 2, curriculum window +30 samples every 80 epochs,
dopri5 at rtol 1e-5 / atol 1e-7, observation noise σ = 0.03 on (x, z).

Metrics for every test orbit, integrating from its exact initial condition
over t ∈ [0, 22]:

* ``rel_err``: √MSE / max|y| in (x, z);
* ``h_drift``: std of the true H along the predicted orbit, divided by
  H(y0) − H_min (the height of that orbit above the fixed point, so that
  orbits with H close to zero do not blow the ratio up).

Usage:
    python -m scripts.ablation_ic --model hnn --regime multi --seed 0
    python -m scripts.ablation_ic --model hnn --regime multi --seed 0 --y0-prior
    python -m scripts.ablation_ic --summarize        # aggregate all runs
    python -m scripts.ablation_ic --figure           # figures/hnn_ablation_ic.png
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from src.hnn import HNN
from src.ode_net import ODEFunc, integrate
from src.systems import LotkaParams, generate_trajectory, lotka_invariant, lotka_volterra

PARAMS = LotkaParams()
TRAIN_ICS = {"single": [(10.0, 5.0)], "multi": [(10.0, 5.0), (6.0, 4.0), (12.0, 6.0)]}
TEST_ICS = [(8.0, 4.0), (14.0, 3.0), (5.0, 3.5), (16.0, 5.0), (11.0, 7.0)]
T_TRAIN, N_TRAIN, T_TEST, N_TEST = 12.0, 300, 22.0, 550
NOISE, EPS = 0.03, 1e-3
X_EQ = np.array([PARAMS.c / PARAMS.d, PARAMS.a / PARAMS.b])        # fixed point (4, 2.75)
H_MIN = float(lotka_invariant(X_EQ, PARAMS))
OUT = Path("data/ablation_ic")
OUT_PRIOR = Path("data/ablation_ic_prior")


def _log_coords(model: str) -> bool:
    return model in ("mlp_log", "hnn")


def _to_model(y: np.ndarray, model: str) -> np.ndarray:
    return np.log(np.clip(y, EPS, None)) if _log_coords(model) else y


def _from_model(u: np.ndarray, model: str) -> np.ndarray:
    return np.exp(u) if _log_coords(model) else u


def noise_scale(obs0: np.ndarray, model: str) -> np.ndarray:
    """Observation noise of y_obs(0) in model coordinates (delta method in log)."""
    return NOISE / np.clip(obs0, EPS, None) if _log_coords(model) else np.full_like(obs0, NOISE)


def train_one(model: str, regime: str, seed: int, epochs: int = 2000,
              y0_prior: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    t_train = np.linspace(0.0, T_TRAIN, N_TRAIN)
    obs = np.stack([
        generate_trajectory(lotka_volterra, np.array(ic), t_train, params=PARAMS,
                            noise=NOISE, seed=1000 * seed + k)
        for k, ic in enumerate(TRAIN_ICS[regime])
    ], axis=1)                                             # (T, B, 2)
    target = torch.tensor(_to_model(obs, model), dtype=torch.float32)

    func = HNN(dim=2, hidden=96, depth=3) if model == "hnn" else ODEFunc(dim=2, hidden=96, depth=3)
    y0 = nn.Parameter(target[0].clone())                   # learned initial states
    y0_obs = target[0].clone()
    s0 = torch.tensor(noise_scale(obs[0], model), dtype=torch.float32)
    opt = optim.Adam(list(func.parameters()) + [y0], lr=2e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t = torch.tensor(t_train, dtype=torch.float32)
    window, history, t0 = 30, [], time.time()
    for epoch in range(epochs):
        if epoch > 0 and epoch % 80 == 0:
            window = min(window + 30, N_TRAIN)
        opt.zero_grad()
        pred = integrate(func, y0, t[:window], rtol=1e-5, atol=1e-7)
        loss = ((pred - target[:window]) ** 2).mean()
        if y0_prior:
            # same units as the MSE: residuals and prior both scaled by s² / N
            n_res = pred.numel()
            loss = loss + (((y0 - y0_obs) / s0) ** 2 * s0.pow(2).mean()).sum() / n_res
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(func.parameters()) + [y0], max_norm=2.0)
        opt.step()
        sched.step()
        history.append(float(loss.detach()))
        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"  [{model}/{regime}/s{seed}] epoch {epoch:4d} win {window:3d} loss {history[-1]:.3e}",
                  flush=True)
    elapsed = time.time() - t0

    y0_true = _to_model(np.array(TRAIN_ICS[regime]), model)
    y0_err = float(np.abs(y0.detach().numpy() - y0_true).max())
    y0_shift_sigma = (np.abs(y0.detach().numpy() - obs[0] if not _log_coords(model)
                             else y0.detach().numpy() - np.log(np.clip(obs[0], EPS, None)))
                      / noise_scale(obs[0], model))

    func.eval()
    t_test = np.linspace(0.0, T_TEST, N_TEST)
    tt = torch.tensor(t_test, dtype=torch.float32)

    def score(ic):
        truth = generate_trajectory(lotka_volterra, np.array(ic), t_test, params=PARAMS)
        u0 = torch.tensor(_to_model(np.array(ic), model), dtype=torch.float32)
        u = integrate(func, u0, tt, rtol=1e-5, atol=1e-7).detach().numpy()
        y = np.clip(_from_model(u, model), EPS, None)
        if not np.all(np.isfinite(y)):
            return {"ic": list(ic), "rel_err": float("nan"), "h_drift": float("nan")}
        h_true = float(lotka_invariant(np.array(ic), PARAMS))
        return {"ic": list(ic), "H0": h_true,
                "rel_err": float(np.sqrt(np.mean((y - truth) ** 2)) / np.abs(truth).max()),
                "h_drift": float(np.std(lotka_invariant(y, PARAMS)) / (h_true - H_MIN))}

    res = {"model": model, "regime": regime, "seed": seed, "epochs": epochs,
           "n_params": sum(p.numel() for p in func.parameters()),
           "elapsed_s": round(elapsed, 1), "final_loss": history[-1],
           "y0_max_abs_error": y0_err, "y0_prior": y0_prior,
           "y0_shift_in_noise_units": y0_shift_sigma.tolist(),
           "train_orbits": [score(ic) for ic in TRAIN_ICS[regime]],
           "test_orbits": [score(ic) for ic in TEST_ICS]}
    out = OUT_PRIOR if y0_prior else OUT
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{model}_{regime}_s{seed}.json", "w") as fh:
        json.dump(res, fh, indent=2)
    torch.save(func.state_dict(), out / f"{model}_{regime}_s{seed}.pt")
    print(f"[ok] {model}/{regime}/s{seed}: {elapsed:.0f} s, test rel_err "
          f"{np.nanmean([o['rel_err'] for o in res['test_orbits']]):.3%}, "
          f"test h_drift {np.nanmean([o['h_drift'] for o in res['test_orbits']]):.3%}", flush=True)
    return res


def summarize(out: str = "data/ablation_ic_summary.json", src: Path = OUT) -> None:
    runs = [json.load(open(f)) for f in sorted(glob.glob(str(src / "*.json")))]
    table = {}
    for r in runs:
        key = f"{r['model']}|{r['regime']}"
        table.setdefault(key, []).append(r)
    summary = {}
    print(f"{'model':8s} {'regime':7s} n  "
          f"{'train-orbit rel_err':>22s} {'train-orbit h_drift':>22s} "
          f"{'test rel_err':>20s} {'test h_drift':>20s}")
    for key, rs in sorted(table.items()):
        def agg(vals):
            v = np.array(vals, dtype=float)
            return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                    "per_seed": v.tolist()}
        tr_rel = agg([r["train_orbits"][0]["rel_err"] for r in rs])
        tr_h = agg([r["train_orbits"][0]["h_drift"] for r in rs])
        te_rel = agg([np.nanmean([o["rel_err"] for o in r["test_orbits"]]) for r in rs])
        te_h = agg([np.nanmean([o["h_drift"] for o in r["test_orbits"]]) for r in rs])
        per_ic = {str(tuple(ic)): {
            "rel_err": agg([r["test_orbits"][i]["rel_err"] for r in rs]),
            "h_drift": agg([r["test_orbits"][i]["h_drift"] for r in rs])}
            for i, ic in enumerate(TEST_ICS)}
        summary[key] = {"n_seeds": len(rs), "train_orbit_rel_err": tr_rel, "train_orbit_h_drift": tr_h,
                        "test_rel_err": te_rel, "test_h_drift": te_h, "test_per_ic": per_ic,
                        "y0_max_abs_error": agg([r["y0_max_abs_error"] for r in rs]),
                        "y0_max_shift_in_noise_units": agg(
                            [np.max(r["y0_shift_in_noise_units"]) if "y0_shift_in_noise_units" in r
                             else np.nan for r in rs]),
                        "elapsed_s": agg([r["elapsed_s"] for r in rs])}
        m, g = key.split("|")
        f = lambda a: f"{a['mean']:.2%} ± {a['std']:.2%}"   # noqa: E731
        print(f"{m:8s} {g:7s} {len(rs)}  {f(tr_rel):>22s} {f(tr_h):>22s} {f(te_rel):>20s} {f(te_h):>20s}")
    with open(out, "w") as fh:
        json.dump({"test_ics": TEST_ICS, "train_ics": TRAIN_ICS, "H_min": H_MIN,
                   "results": summary}, fh, indent=2)
    print(f"[ok] {out}")


def figure(summary_path: str = "data/ablation_ic_summary.json",
           out: str = "figures/hnn_ablation_ic.png", seed: int = 0) -> None:
    """(a) relative error and (b) H drift on the held-out orbits, every seed
    as a dot over the mean bar; (c) the held-out orbits predicted by the HNN
    and by the log-coordinate MLP trained on three trajectories (one seed)."""
    import matplotlib.pyplot as plt

    from src.plots import _save, _style

    _style()

    S = json.load(open(summary_path))["results"]
    models = [("mlp", "MLP (x, z)"), ("mlp_log", "MLP (ln x, ln z)"), ("hnn", "HNN")]
    colors = {"mlp": "#9aa3b8", "mlp_log": "#7c5cff", "hnn": "#f472b6"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, metric, title in [(axes[0], "test_rel_err", "(a) relative error, unseen orbits"),
                              (axes[1], "test_h_drift", "(b) drift of the true H, unseen orbits")]:
        for gi, regime in enumerate(["single", "multi"]):
            for mi, (m, label) in enumerate(models):
                a = S[f"{m}|{regime}"][metric]
                x = gi * 4 + mi
                ax.bar(x, 100 * a["mean"], color=colors[m], alpha=0.75,
                       label=label if gi == 0 else None)
                ax.scatter(np.full(len(a["per_seed"]), x), 100 * np.array(a["per_seed"]),
                           color="white", s=12, zorder=3)
        ax.set_xticks([1, 5], ["1 training trajectory", "3 training trajectories"])
        ax.set_ylabel("%")
        ax.set_yscale("log")
        ticks = [1, 2, 5, 10, 20, 50, 100]
        ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
        ax.minorticks_off()
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=9, loc="upper right")

    ax = axes[2]
    t_test = np.linspace(0.0, T_TEST, N_TEST)
    tt = torch.tensor(t_test, dtype=torch.float32)
    for k, ic in enumerate(TEST_ICS):
        truth = generate_trajectory(lotka_volterra, np.array(ic), t_test, params=PARAMS)
        ax.plot(truth[:, 0], truth[:, 1], color="#22d3ee", lw=1.2, alpha=0.9,
                label="truth" if k == 0 else None)
    for m, style in [("mlp_log", "--"), ("hnn", "-")]:
        func = HNN(2, 96, 3) if m == "hnn" else ODEFunc(2, 96, 3)
        func.load_state_dict(torch.load(OUT / f"{m}_multi_s{seed}.pt"))
        func.eval()
        for k, ic in enumerate(TEST_ICS):
            u = integrate(func, torch.tensor(np.log(ic), dtype=torch.float32), tt).detach().numpy()
            y = np.exp(u)
            ax.plot(y[:, 0], y[:, 1], style, color=colors[m], lw=1.0,
                    label=dict(models)[m] if k == 0 else None)
    for k, ic in enumerate(TRAIN_ICS["multi"]):
        tr = generate_trajectory(lotka_volterra, np.array(ic), t_test, params=PARAMS)
        ax.plot(tr[:, 0], tr[:, 1], color="white", lw=0.6, alpha=0.35,
                label="training orbits" if k == 0 else None)
    ax.set(xlabel="prey x", ylabel="predator z")
    ax.set_title(f"(c) unseen orbits, 3-trajectory training, seed {seed}", loc="left")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    _save(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["mlp", "mlp_log", "hnn"])
    p.add_argument("--regime", choices=["single", "multi"])
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--figure", action="store_true")
    p.add_argument("--y0-prior", action="store_true",
                   help="Gaussian prior on the learned initial state (runs in data/ablation_ic_prior/)")
    a = p.parse_args()
    if a.summarize and a.y0_prior:
        summarize("data/ablation_ic_prior_summary.json", OUT_PRIOR)
    elif a.summarize:
        summarize()
    elif a.figure:
        figure()
    else:
        for s in a.seed:
            train_one(a.model, a.regime, s, a.epochs, a.y0_prior)
