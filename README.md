# Neural ODEs para sistemas dinámicos

Aprender el campo vectorial de un sistema dinámico (péndulo amortiguado, Lotka-Volterra) desde trayectorias observadas, usando Neural ODEs (Chen et al., 2018).

## Motivación

Las redes neuronales clásicas aprenden funciones directas `x → y`. Las Neural ODEs aprenden la **derivada** `dy/dt = f_θ(y, t)` y la integran con un solver numérico. Esto permite:

- Tratar el "número de capas" como un hiperparámetro continuo (tiempo de integración).
- Respetar conservaciones físicas con la estructura correcta.
- Extrapolar dinámica fuera del rango de entrenamiento.

Este proyecto aprovecha el background en EDOs: la arquitectura es natural de entender.

## Stack

- Python 3.11+
- PyTorch 2.x
- torchdiffeq (integradores diferenciables)
- Matplotlib

## Estructura

```
02-neural-odes/
├── README.md
├── requirements.txt
├── src/
│   ├── systems.py        # Sistemas de referencia (péndulo, Lotka-Volterra)
│   ├── ode_net.py        # Red neuronal que aproxima f(y, t)
│   └── train.py          # Training loop + evaluación
└── notebooks/
```

## Roadmap

- [ ] Implementar sistemas de referencia y generar trayectorias sintéticas con ruido.
- [ ] Definir `ODEFunc` (MLP que aproxima el campo vectorial).
- [ ] Training loop con `odeint` de torchdiffeq.
- [ ] Evaluar extrapolación más allá del horizonte de entrenamiento.
- [ ] GIF comparando trayectoria real vs. trayectoria predicha.

## Resultado esperado

Una red de ~3 capas MLP, entrenada con trayectorias de 5 segundos, debería extrapolar correctamente a 10+ segundos, conservando aproximadamente la energía del sistema.
