"""Sistemas dinámicos de referencia para generar trayectorias sintéticas."""

from __future__ import annotations

import numpy as np


def damped_pendulum(y: np.ndarray, t: float, gamma: float = 0.1, omega0: float = 1.0) -> np.ndarray:
    theta, theta_dot = y
    return np.array([theta_dot, -omega0**2 * np.sin(theta) - gamma * theta_dot])


def lotka_volterra(y: np.ndarray, t: float, a: float = 1.0, b: float = 0.4, c: float = 0.4, d: float = 0.1) -> np.ndarray:
    x, z = y
    return np.array([a * x - b * x * z, d * x * z - c * z])


def generate_trajectory(system, y0: np.ndarray, t: np.ndarray, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Integra el sistema con scipy.integrate.odeint y opcionalmente agrega ruido gaussiano."""
    from scipy.integrate import odeint

    traj = odeint(system, y0, t)
    if noise > 0:
        rng = np.random.default_rng(seed)
        traj = traj + rng.normal(0.0, noise, size=traj.shape)
    return traj
