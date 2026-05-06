"""Tests for the Neural ODE and Hamiltonian Neural Network architectures.

The HNN's defining property is that integrating its symplectic field
preserves the learned scalar ``H_theta`` exactly, even at random
initialisation. The conservation test below exercises that guarantee on
an untrained network.
"""
from __future__ import annotations

import torch

from src.hnn import HNN, count_parameters as count_params_hnn
from src.ode_net import ODEFunc, count_parameters, integrate


def test_odefunc_forward_shape():
    func = ODEFunc(dim=2, hidden=16, depth=2)
    y = torch.randn(4, 2)
    out = func(torch.tensor(0.0), y)
    assert out.shape == y.shape


def test_odefunc_integrate_shape():
    func = ODEFunc(dim=2, hidden=16, depth=2)
    y0 = torch.zeros(2)
    t = torch.linspace(0, 1, 10)
    traj = integrate(func, y0, t)
    assert traj.shape == (10, 2)


def test_hnn_forward_shape():
    hnn = HNN(dim=2, hidden=32, depth=2)
    y = torch.randn(4, 2)
    out = hnn(torch.tensor(0.0), y)
    assert out.shape == y.shape


def test_hnn_only_supports_2d():
    import pytest
    with pytest.raises(NotImplementedError):
        HNN(dim=3, hidden=16, depth=2)


def test_hnn_conserves_hamiltonian_along_trajectory():
    """Integrating an untrained HNN's field must conserve H_theta to integrator tolerance."""
    torch.manual_seed(0)
    hnn = HNN(dim=2, hidden=32, depth=2)
    y0 = torch.tensor([0.5, -0.3])
    t = torch.linspace(0, 1.0, 20)
    traj = integrate(hnn, y0, t, rtol=1e-7, atol=1e-9)
    with torch.no_grad():
        H = hnn.hamiltonian(traj)
    drift = (H.max() - H.min()).item()
    assert drift < 1e-3, f"H drifted by {drift:.2e}"


def test_count_parameters_positive():
    assert count_parameters(ODEFunc(dim=2, hidden=16, depth=2)) > 0
    assert count_params_hnn(HNN(dim=2, hidden=16, depth=2)) > 0
