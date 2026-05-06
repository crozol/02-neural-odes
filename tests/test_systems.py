"""Tests for the reference dynamical systems.

The two systems expose different invariants (energy decay for the damped
pendulum, conserved orbit for Lotka-Volterra) and the tests use those exact
invariants as physical sanity checks on the trajectory generator.
"""
from __future__ import annotations

import numpy as np

from src.systems import (
    damped_pendulum,
    generate_trajectory,
    lotka_invariant,
    lotka_volterra,
    pendulum_energy,
)


def test_pendulum_rhs_shape():
    rhs = damped_pendulum(np.array([0.5, 0.1]), 0.0)
    assert rhs.shape == (2,)


def test_lotka_rhs_shape():
    rhs = lotka_volterra(np.array([1.0, 0.5]), 0.0)
    assert rhs.shape == (2,)


def test_generate_trajectory_shape_and_initial_condition():
    y0 = np.array([1.0, 0.0])
    t = np.linspace(0, 5, 100)
    traj = generate_trajectory(damped_pendulum, y0, t)
    assert traj.shape == (100, 2)
    np.testing.assert_allclose(traj[0], y0, atol=1e-6)


def test_damped_pendulum_energy_decays():
    """Pendulum energy under friction must be non-increasing along the trajectory."""
    y0 = np.array([1.0, 0.0])
    t = np.linspace(0, 10, 200)
    traj = generate_trajectory(damped_pendulum, y0, t)
    energies = pendulum_energy(traj)
    diffs = np.diff(energies)
    # tiny positive deltas can come from the integrator; the bound is generous
    assert (diffs <= 1e-6).all(), f"max energy increase = {diffs.max():.2e}"


def test_lotka_invariant_conserved():
    """The Lotka-Volterra invariant should stay nearly constant along an orbit."""
    y0 = np.array([1.0, 0.5])
    t = np.linspace(0, 10, 200)
    traj = generate_trajectory(lotka_volterra, y0, t)
    H = lotka_invariant(traj)
    rel_drift = (H.max() - H.min()) / abs(H.mean())
    assert rel_drift < 1e-3, f"relative drift = {rel_drift:.2e}"
