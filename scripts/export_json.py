"""Export trained-model arrays + metrics to JSON for the portfolio site.

The output JSON is consumed by Plotly charts on the project detail page
(``website/projects/02-neural-odes.html``).

Usage:
    python -m scripts.export_json
    python -m scripts.export_json --out ../website/assets/data/02-neural-odes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _thin(arr: np.ndarray, max_points: int) -> np.ndarray:
    """Subsample along the leading axis to cap file size."""
    if arr.shape[0] <= max_points:
        return arr
    step = max(1, arr.shape[0] // max_points)
    return arr[::step]


def _load_metrics(path: Path) -> dict:
    with open(path) as f:
        summary = json.load(f)
    out = {}
    for exp in summary.get("experiments", []):
        out[exp["system"]] = exp
    return out


def _system_payload(npz_path: Path, *, max_traj: int = 400, max_loss: int = 1200,
                    max_vf: int = 24) -> dict:
    z = np.load(npz_path)
    t = _thin(z["t"], max_traj)
    y_true = _thin(z["y_true"], max_traj)
    y_pred = _thin(z["y_pred"], max_traj)

    payload = {
        "t": t.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "loss_history": _thin(z["loss_history"], max_loss).tolist(),
        "t_train_max": float(z["t_train_max"]),
        "t_full_max": float(z["t_full_max"]),
        "grow_every": int(z["grow_every"]),
        "noise_obs": float(z["noise_obs"]),
    }
    if "vf_X" in z.files:
        payload["vf"] = {
            "X": z["vf_X"].tolist(),
            "Y": z["vf_Y"].tolist(),
            "U": z["vf_U"].tolist(),
            "V": z["vf_V"].tolist(),
        }

    if "energy_true" in z.files:
        payload["energy_true"] = _thin(z["energy_true"], max_traj).tolist()
        payload["energy_pred"] = _thin(z["energy_pred"], max_traj).tolist()
    if "invariant_true" in z.files:
        payload["invariant_true"] = _thin(z["invariant_true"], max_traj).tolist()
        payload["invariant_pred"] = _thin(z["invariant_pred"], max_traj).tolist()

    return payload


def main(out_path: str = "../website/assets/data/02-neural-odes.json",
         data_dir: str = "data") -> None:
    data_root = Path(data_dir)

    metrics = _load_metrics(data_root / "metrics.json")
    pendulum = _system_payload(data_root / "pendulum_arrays.npz")
    lotka = _system_payload(data_root / "lotka_arrays.npz")

    bundle = {
        "metrics": metrics,
        "pendulum": pendulum,
        "lotka": lotka,
    }

    hnn_path = data_root / "lotka_hnn_arrays.npz"
    if hnn_path.exists():
        bundle["lotka_hnn"] = _system_payload(hnn_path)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))

    size_kb = out.stat().st_size / 1024
    print(f"[ok] {out}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="../website/assets/data/02-neural-odes.json")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    main(out_path=args.out, data_dir=args.data_dir)
