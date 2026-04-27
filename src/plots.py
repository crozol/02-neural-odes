"""Publication-style static figures for the README and the portfolio page,
matching the dark theme used in the rest of the portfolio.

Five PNGs are written to the output directory:

    1. loss.png             — training loss vs epoch (log-scale)
    2. pendulum_traj.png    — time series with extrapolation region shaded
    3. pendulum_phase.png   — phase plane: real vs predicted orbit + learned
                              vector field
    4. pendulum_energy.png  — mechanical energy decay (true vs learned)
    5. lotka_phase.png      — Lotka-Volterra closed orbit in (prey, predator)
                              space, real vs predicted

Design rules mirror those of ``01-ising-bayesian/src/plots.py``: dark panel,
muted grid, monospace for numbers, sans-serif for prose. Palette matches the
website CSS variables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------- portfolio palette ------------------------------ #

BG_PANEL = "#0d1220"
BG_AXES = "#0c101c"
FG_0 = "#e7ecf5"
FG_1 = "#9aa3b8"
GRID = (1.0, 1.0, 1.0, 0.05)
SPINE = (1.0, 1.0, 1.0, 0.20)

PURPLE = "#7c5cff"   # network / learned
CYAN = "#22d3ee"     # ground truth
PINK = "#f472b6"     # highlight / posterior-equivalent
AMBER = "#fbbf24"    # exact / boundary marker

SANS = ["DejaVu Sans", "Inter", "Segoe UI", "Arial"]
MONO = ["DejaVu Sans Mono", "JetBrains Mono", "Consolas", "monospace"]


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": BG_PANEL,
        "savefig.facecolor": BG_PANEL,
        "axes.facecolor": BG_AXES,
        "axes.edgecolor": SPINE,
        "axes.labelcolor": FG_0,
        "axes.titlecolor": FG_0,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": FG_1,
        "ytick.color": FG_1,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "text.color": FG_0,
        "legend.frameon": True,
        "legend.facecolor": BG_PANEL,
        "legend.edgecolor": SPINE,
        "legend.labelcolor": FG_0,
        "legend.fontsize": 10,
        "font.family": SANS,
        "mathtext.fontset": "cm",
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
    })


def _save(fig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG_PANEL)
    print(f"[ok] {path}")


# --------------------------------------------------------------------------- #
#  Figure 1 · training loss
# --------------------------------------------------------------------------- #

def plot_loss(history, out_path: str, *, system_name: str = "pendulum",
              grow_every: int | None = None) -> None:
    import matplotlib.pyplot as plt

    _style()
    history = np.asarray(history)
    epochs = np.arange(len(history))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.patch.set_facecolor(BG_PANEL)

    ax.plot(epochs, history, color=PURPLE, lw=1.4, alpha=0.85)
    ax.set_yscale("log")

    if grow_every is not None and grow_every > 0:
        for x in range(grow_every, len(history), grow_every):
            ax.axvline(x, color=AMBER, lw=0.6, alpha=0.35, ls="--")
        ax.text(0.012, 0.05,
                f"vertical ticks · curriculum window growth (every {grow_every} epochs)",
                transform=ax.transAxes, fontsize=9.5, color=FG_1, family=MONO,
                ha="left", va="bottom", style="italic")

    final_loss = float(history[-1])
    ax.text(0.99, 0.95, fr"final  MSE = {final_loss:.2e}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color=FG_0, family=MONO, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                      edgecolor=PURPLE, linewidth=0.9, alpha=0.95))

    ax.set_xlabel("epoch")
    ax.set_ylabel("training MSE  (log scale)")
    ax.set_title(f"Training loss · {system_name}", loc="left", pad=10)

    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 2 · time-series trajectories
# --------------------------------------------------------------------------- #

def plot_time_series(t, y_true, y_pred, t_train_max, out_path: str,
                     *, labels=("θ  (rad)", "θ̇  (rad/s)"),
                     system_name: str = "damped pendulum") -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.4), sharex=True)
    fig.patch.set_facecolor(BG_PANEL)

    for i, ax in enumerate(axes):
        ax.axvspan(t_train_max, t.max(), color=AMBER, alpha=0.06, zorder=0,
                   label="extrapolation region" if i == 0 else None)
        ax.plot(t, y_true[:, i], color=CYAN, lw=2.0, label="ground truth"
                if i == 0 else None)
        ax.plot(t, y_pred[:, i], color=PURPLE, lw=2.0, ls="--",
                label="Neural ODE" if i == 0 else None)
        ax.axvline(t_train_max, color=AMBER, lw=1.4, ls="--", alpha=0.7)
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("time  (s)")
    axes[0].set_title(
        f"Trajectory reconstruction · {system_name}  ·  trained on t ≤ {t_train_max:.1f} s",
        loc="left", pad=10,
    )

    handles, lbls = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc="lower center", ncol=len(handles),
                   fontsize=10.5, framealpha=0.96,
                   bbox_to_anchor=(0.5, -0.01), handlelength=2.0)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 3 · phase plane + learned vector field
# --------------------------------------------------------------------------- #

def plot_phase(y_true, y_pred, vector_field, out_path: str,
               *, axis_labels=("θ", "θ̇"),
               title: str = "Phase plane · damped pendulum") -> None:
    """Real vs predicted orbit overlaid on the *learned* vector field."""
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    fig.patch.set_facecolor(BG_PANEL)

    X, Y, U, V = vector_field
    speed = np.sqrt(U.numpy() ** 2 + V.numpy() ** 2)
    ax.streamplot(
        X.numpy(), Y.numpy(), U.numpy(), V.numpy(),
        color=speed, cmap="magma", density=1.2, linewidth=0.85,
        arrowsize=1.0,
    )

    ax.plot(y_true[:, 0], y_true[:, 1], color=CYAN, lw=2.4,
            label="ground-truth orbit", zorder=4)
    ax.plot(y_pred[:, 0], y_pred[:, 1], color=PURPLE, lw=2.0, ls="--",
            label="Neural ODE orbit", zorder=5)
    ax.scatter([y_true[0, 0]], [y_true[0, 1]], color=AMBER, s=70, zorder=6,
               edgecolor=BG_PANEL, linewidth=1.5, label="initial condition")

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="upper right", framealpha=0.96, fontsize=9.8)

    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 4 · energy decay
# --------------------------------------------------------------------------- #

def plot_energy(t, energy_true, energy_pred, t_train_max, out_path: str,
                *, ylabel: str = "mechanical energy",
                title: str = "Energy decay · damped pendulum") -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor(BG_PANEL)

    ax.axvspan(t_train_max, t.max(), color=AMBER, alpha=0.06, zorder=0)
    ax.plot(t, energy_true, color=CYAN, lw=2.2, label="ground truth")
    ax.plot(t, energy_pred, color=PURPLE, lw=2.0, ls="--", label="Neural ODE")
    ax.axvline(t_train_max, color=AMBER, lw=1.4, ls="--", alpha=0.7)

    drift = float(np.abs(energy_pred[-1] - energy_true[-1]) /
                  (np.abs(energy_true[-1]) + 1e-12))
    ax.text(0.012, 0.05,
            fr"final  |dE / E| = {drift:.2%}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, color=FG_0, family=MONO, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                      edgecolor=PURPLE, linewidth=0.9, alpha=0.95))

    ax.set_xlabel("time  (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="upper right", framealpha=0.96, fontsize=10)

    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 5 · Lotka-Volterra phase orbit
# --------------------------------------------------------------------------- #

def plot_lotka_phase(y_true, y_pred, vector_field, out_path: str) -> None:
    plot_phase(
        y_true, y_pred, vector_field, out_path,
        axis_labels=("prey  x", "predator  z"),
        title="Phase plane · Lotka-Volterra",
    )


def plot_lotka_time(t, y_true, y_pred, t_train_max, out_path: str) -> None:
    plot_time_series(
        t, y_true, y_pred, t_train_max, out_path,
        labels=("prey  x", "predator  z"),
        system_name="Lotka-Volterra",
    )
