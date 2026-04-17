# Neural ODEs for dynamical systems

Learning the vector field of a dynamical system (damped pendulum, Lotka-Volterra) from observed trajectories, using Neural ODEs (Chen et al., 2018).

## Motivation

Classic neural networks learn direct functions `x → y`. **Neural ODEs** learn the **derivative** `dy/dt = f_θ(y, t)` and integrate it with a numerical solver. This allows:

- Treating "number of layers" as a continuous hyperparameter (integration time).
- Respecting physical conservation laws through architectural structure.
- Extrapolating dynamics beyond the training range.

This project leverages the physics-mathematics background: if you understand ODEs well, the architecture is intuitive.

## Stack

- Python 3.11+
- PyTorch 2.x
- torchdiffeq (differentiable integrators)
- Matplotlib

## Structure

```
02-neural-odes/
├── README.md
├── requirements.txt
├── src/
│   ├── systems.py        # Reference systems (pendulum, Lotka-Volterra)
│   ├── ode_net.py        # Neural network approximating f(y, t)
│   └── train.py          # Training loop + evaluation
└── notebooks/
```

## Roadmap

- [ ] Implement reference systems and generate synthetic trajectories with noise.
- [ ] Define `ODEFunc` (MLP approximating the vector field).
- [ ] Training loop with torchdiffeq's `odeint`.
- [ ] Evaluate extrapolation beyond the training horizon.
- [ ] GIF comparing real vs. predicted trajectory.

## Expected results

A ~3-layer MLP, trained with 5-second trajectories, should extrapolate correctly to 10+ seconds, approximately conserving the system's energy.
