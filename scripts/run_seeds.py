"""Run the MLP-baseline pipeline ``main.py`` across a small ensemble of seeds.

Each metric reported by ``run_pendulum`` and ``run_lotka`` is a single-seed
quantity in ``main.py``; a comité (or any rigorous reader) wants to know
whether those numbers reproduce. This script wraps both experiments in an
outer seed loop and aggregates the resulting metrics into
``data/seed_sweep_metrics.json`` as ``mean ± std`` over ``n_seeds`` runs.

By design the script does *not* touch the per-run ``data/metrics.json`` that
``main.py`` and the HNN ablation scripts write — the aggregated results live
in their own file so the headline figures and the HNN entries are preserved.

Only the **MLP baseline** is swept (pendulum + Lotka). The HNN ablation is
trained once with its canonical seed because it carries its own structural
prior; rerunning it across seeds would multiply compute time without
materially changing the architectural conclusion.

Usage:
    python -m scripts.run_seeds
    python -m scripts.run_seeds --seeds 0 1 2 3 4
    python -m scripts.run_seeds --n-seeds 5     # equivalent to seeds 0..4

Output:
    data/seed_sweep_metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from main import run_lotka, run_pendulum


def _aggregate(runs: list[dict]) -> dict:
    """Compute mean / std / min / max for every numeric field shared across runs."""
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    out: dict = {}
    for k in keys:
        values = np.array([r[k] for r in runs], dtype=float)
        out[k] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def _parse_seeds(argv: argparse.Namespace) -> list[int]:
    if argv.seeds is not None:
        return list(argv.seeds)
    return list(range(argv.n_seeds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit list of seeds, e.g. --seeds 0 1 2 3 4")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds (0..n-1) if --seeds is not given")
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    seeds = _parse_seeds(args)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True, exist_ok=True)

    pendulum_runs: list[dict] = []
    lotka_runs: list[dict] = []

    print("=" * 64)
    print(f"SEED SWEEP · MLP baseline · seeds = {seeds}")
    print("=" * 64)

    t0_total = time.time()
    for s in seeds:
        print(f"\n----- seed = {s} -----")
        pendulum_runs.append(run_pendulum(out_dir, data_dir, s))
        lotka_runs.append(run_lotka(out_dir, data_dir, s))

    elapsed_total = time.time() - t0_total

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "total_elapsed_s": round(elapsed_total, 2),
        "damped_pendulum": {
            "per_seed": pendulum_runs,
            "aggregated": _aggregate(pendulum_runs),
        },
        "lotka_volterra": {
            "per_seed": lotka_runs,
            "aggregated": _aggregate(lotka_runs),
        },
    }

    out_path = data_dir / "seed_sweep_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print a compact summary to stdout for the README author
    print("\n" + "=" * 64)
    print("AGGREGATED  (mean ± std over n_seeds runs)")
    print("=" * 64)
    for name, runs, agg in (
        ("damped_pendulum", pendulum_runs, summary["damped_pendulum"]["aggregated"]),
        ("lotka_volterra", lotka_runs, summary["lotka_volterra"]["aggregated"]),
    ):
        m = agg["mse_train"]["mean"];      s = agg["mse_train"]["std"]
        print(f"{name:18s}  mse_train       = {m:.3e} ± {s:.1e}")
        m = agg["mse_extrap"]["mean"];     s = agg["mse_extrap"]["std"]
        print(f"{name:18s}  mse_extrap      = {m:.3e} ± {s:.1e}")
        m = agg["rel_err_extrap"]["mean"]; s = agg["rel_err_extrap"]["std"]
        print(f"{name:18s}  rel_err_extrap  = {m:.3%} ± {s:.2%}")
        if "energy_drift_final" in agg:
            m = agg["energy_drift_final"]["mean"]; s = agg["energy_drift_final"]["std"]
            print(f"{name:18s}  energy_drift    = {m:.2%} ± {s:.2%}")
        if "invariant_drift_relative" in agg:
            m = agg["invariant_drift_relative"]["mean"]
            s = agg["invariant_drift_relative"]["std"]
            print(f"{name:18s}  invariant_drift = {m:.2%} ± {s:.2%}")
        print()

    print(f"[ok] {out_path}")
    print(f"[ok] total elapsed: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
