"""Reference dynamical systems used as ground truth for the Neural ODE.

Two systems are exposed, both 2D so they admit a clean phase-plane view:

* ``damped_pendulum`` — second-order ODE rewritten as a 2D first-order system
  ``(theta, theta_dot)``. Energy decays monotonically because of the friction
  term, which gives a non-trivial check (the learned field should also dissipate
  energy).
* ``lotka_volterra`` — predator-prey system with closed orbits in phase space.
  A textbook example of a conservative-like flow that a memoryless MLP cannot
  reproduce, but a Neural ODE can.

Trajectories are integrated with ``scipy.integrate.odeint`` (LSODA) at a fixed
time grid and Gaussian observation noise can be added to imitate measurement
error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint


# --------------------------------------------------------------------------- #
#  systems
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PendulumParams:
    gamma: float = 0.15      # damping coefficient
    omega0: float = 1.5      # natural angular frequency


@dataclass(frozen=True)
class LotkaParams:
    a: float = 1.1           # prey growth rate
    b: float = 0.4           # predation rate
    c: float = 0.4           # predator death rate
    d: float = 0.1           # predator efficiency


def damped_pendulum(y: np.ndarray, t: float,
                    params: PendulumParams = PendulumParams()) -> np.ndarray:
    """Right-hand side of theta_ddot + gamma*theta_dot + omega0^2 sin theta = 0."""
    theta, theta_dot = y
    return np.array([
        theta_dot,
        -params.omega0 ** 2 * np.sin(theta) - params.gamma * theta_dot,
    ])


def lotka_volterra(y: np.ndarray, t: float,
                   params: LotkaParams = LotkaParams()) -> np.ndarray:
    """Right-hand side of the classical Lotka-Volterra predator-prey ODE."""
    x, z = y
    return np.array([
        params.a * x - params.b * x * z,
        params.d * x * z - params.c * z,
    ])


# --------------------------------------------------------------------------- #
#  trajectory generation
# --------------------------------------------------------------------------- #

def generate_trajectory(system, y0: np.ndarray, t: np.ndarray, *,
                        params=None, noise: float = 0.0,
                        seed: int = 0) -> np.ndarray:
    """Integrate ``system`` from ``y0`` over time grid ``t`` and optionally
    add zero-mean Gaussian noise to every state component."""
    if params is None:
        traj = odeint(system, y0, t, rtol=1e-8, atol=1e-10)
    else:
        traj = odeint(system, y0, t, args=(params,), rtol=1e-8, atol=1e-10)
    if noise > 0:
        rng = np.random.default_rng(seed)
        traj = traj + rng.normal(0.0, noise, size=traj.shape)
    return traj


# --------------------------------------------------------------------------- #
#  invariants — used as physical sanity checks on the learned dynamics
# --------------------------------------------------------------------------- #

def pendulum_energy(y: np.ndarray, params: PendulumParams = PendulumParams()) -> np.ndarray:
    """Total mechanical energy per unit mass of the (undamped) pendulum.

    For the *damped* pendulum this is no longer conserved; it should decay
    monotonically over time. The decay rate is itself a structural invariant
    that the learned field must respect.
    """
    theta, theta_dot = y[..., 0], y[..., 1]
    kinetic = 0.5 * theta_dot ** 2
    potential = params.omega0 ** 2 * (1.0 - np.cos(theta))
    return kinetic + potential


def lotka_invariant(y: np.ndarray, params: LotkaParams = LotkaParams()) -> np.ndarray:
    """Conserved quantity of the undamped Lotka-Volterra system.

    ``H(x, z) = d*x - c*ln(x) + b*z - a*ln(z)`` is constant along orbits.
    """
    x, z = y[..., 0], y[..., 1]
    return params.d * x - params.c * np.log(x) + params.b * z - params.a * np.log(z)
