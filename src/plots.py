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

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    fig.patch.set_facecolor(BG_PANEL)

    ax.plot(epochs, history, color=PURPLE, lw=1.6, alpha=0.92,
            label="training MSE")
    ax.set_yscale("log")

    if grow_every is not None and grow_every > 0:
        first = True
        for x in range(grow_every, len(history), grow_every):
            ax.axvline(x, color=AMBER, lw=0.7, alpha=0.45, ls="--",
                       label="curriculum window grows" if first else None)
            first = False

    initial_loss = float(history[0])
    final_loss = float(history[-1])
    best_loss = float(history.min())
    reduction = initial_loss / max(final_loss, 1e-12)

    info = (
        f"initial MSE = {initial_loss:.2e}\n"
        f"final   MSE = {final_loss:.2e}\n"
        f"best    MSE = {best_loss:.2e}\n"
        f"reduction   = {reduction:,.0f}×"
    )
    ax.text(0.985, 0.96, info,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10.5, color=FG_0, family=MONO,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG_PANEL,
                      edgecolor=PURPLE, linewidth=0.9, alpha=0.96))

    ax.set_xlabel("epoch")
    ax.set_ylabel("training MSE  (log scale)")
    ax.set_title(f"Training loss · {system_name}", loc="left", pad=10)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=10)

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
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.6), sharex=True)
    fig.patch.set_facecolor(BG_PANEL)

    train_mask = t <= t_train_max
    extrap_mask = ~train_mask
    train_residuals = float(np.sqrt(np.mean((y_pred[train_mask] - y_true[train_mask]) ** 2)))
    extrap_residuals = float(np.sqrt(np.mean((y_pred[extrap_mask] - y_true[extrap_mask]) ** 2)))

    for i, ax in enumerate(axes):
        ax.axvspan(t_train_max, t.max(), color=AMBER, alpha=0.07, zorder=0,
                   label="extrapolation region" if i == 0 else None)
        ax.plot(t, y_true[:, i], color=CYAN, lw=2.1, label="ground truth"
                if i == 0 else None, zorder=3)
        ax.plot(t, y_pred[:, i], color=PURPLE, lw=2.0, ls="--",
                label="Neural ODE" if i == 0 else None, zorder=4)
        ax.fill_between(t, y_true[:, i], y_pred[:, i],
                        color=PINK, alpha=0.10, zorder=2,
                        label="residual" if i == 0 else None)
        ax.axvline(t_train_max, color=AMBER, lw=1.5, ls="--", alpha=0.8)
        ax.set_ylabel(labels[i])

    axes[0].text(t_train_max, axes[0].get_ylim()[1],
                 f"  end of training  ·  t = {t_train_max:.1f}",
                 ha="left", va="top",
                 fontsize=10, color=AMBER, family=MONO,
                 bbox=dict(boxstyle="round,pad=0.32", facecolor=BG_PANEL,
                           edgecolor=AMBER, linewidth=0.7, alpha=0.95))

    info = (
        f"RMSE  ·  train     = {train_residuals:.3e}\n"
        f"RMSE  ·  extrapol. = {extrap_residuals:.3e}"
    )
    axes[1].text(0.012, 0.05, info,
                 transform=axes[1].transAxes, ha="left", va="bottom",
                 fontsize=10.5, color=FG_0, family=MONO,
                 bbox=dict(boxstyle="round,pad=0.45", facecolor=BG_PANEL,
                           edgecolor=PURPLE, linewidth=0.9, alpha=0.95))

    axes[-1].set_xlabel("time  (s)")
    axes[0].set_title(
        f"Trajectory reconstruction · {system_name}  ·  trained on t ≤ {t_train_max:.1f}",
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
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    fig.patch.set_facecolor(BG_PANEL)

    X, Y, U, V = vector_field
    speed = np.sqrt(U.numpy() ** 2 + V.numpy() ** 2)
    strm = ax.streamplot(
        X.numpy(), Y.numpy(), U.numpy(), V.numpy(),
        color=speed, cmap="magma", density=1.25, linewidth=0.9,
        arrowsize=1.05,
    )

    ax.plot(y_true[:, 0], y_true[:, 1], color=CYAN, lw=2.5,
            label="ground-truth orbit", zorder=4, alpha=0.95)
    ax.plot(y_pred[:, 0], y_pred[:, 1], color=PURPLE, lw=2.0, ls="--",
            label="Neural ODE orbit", zorder=5, alpha=0.95)

    ax.scatter([y_true[0, 0]], [y_true[0, 1]], color=AMBER, s=110, zorder=7,
               edgecolor=BG_PANEL, linewidth=1.8, label="initial state  t = 0",
               marker="o")
    ax.scatter([y_true[-1, 0]], [y_true[-1, 1]], color=CYAN, s=70, zorder=7,
               edgecolor=BG_PANEL, linewidth=1.5, marker="s",
               label="ground-truth final")
    ax.scatter([y_pred[-1, 0]], [y_pred[-1, 1]], color=PURPLE, s=70, zorder=7,
               edgecolor=BG_PANEL, linewidth=1.5, marker="D",
               label="Neural ODE final")

    cbar = fig.colorbar(strm.lines, ax=ax, pad=0.02, shrink=0.88, aspect=28)
    cbar.set_label("learned speed  ‖f$_θ$‖", color=FG_1)
    cbar.ax.yaxis.set_tick_params(color=FG_1, labelcolor=FG_1)
    cbar.outline.set_edgecolor(SPINE)

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
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    fig.patch.set_facecolor(BG_PANEL)

    ax.axvspan(t_train_max, t.max(), color=AMBER, alpha=0.07, zorder=0,
               label="extrapolation region")
    ax.fill_between(t, energy_true, energy_pred,
                    color=PINK, alpha=0.13, zorder=1,
                    label="energy gap")
    ax.plot(t, energy_true, color=CYAN, lw=2.3, label="ground truth", zorder=3)
    ax.plot(t, energy_pred, color=PURPLE, lw=2.0, ls="--",
            label="Neural ODE", zorder=4)
    ax.axvline(t_train_max, color=AMBER, lw=1.5, ls="--", alpha=0.8)

    drift_final = float(np.abs(energy_pred[-1] - energy_true[-1]) /
                        (np.abs(energy_true[-1]) + 1e-12))
    train_mask = t <= t_train_max
    drift_train_mean = float(np.mean(np.abs(energy_pred[train_mask] - energy_true[train_mask]) /
                                     (np.abs(energy_true[train_mask]) + 1e-12)))

    info = (
        f"|ΔE/E|   ·  train mean = {drift_train_mean:.2%}\n"
        f"|ΔE/E|   ·  final t    = {drift_final:.2%}"
    )
    ax.text(0.012, 0.05, info,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.5, color=FG_0, family=MONO,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG_PANEL,
                      edgecolor=PURPLE, linewidth=0.9, alpha=0.95))

    ax.set_xlabel("time  (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="upper right", framealpha=0.96, fontsize=10, ncol=2)

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
