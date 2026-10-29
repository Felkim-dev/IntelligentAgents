"""Discretizador uniforme: observación continua -> tupla de índices de bin."""
import numpy as np

# Rango fijo por dimensión de CartPole: x, x_dot, theta, theta_dot.
# x y theta usan los límites de terminación (±2.4 m, ±12°), así ningún bin se gasta en estados
# donde el episodio ya terminó. Las velocidades no tienen límite: se recortan al rango fijo.
CARTPOLE_LOW = np.array([-2.4, -3.0, -0.2095, -3.5])
CARTPOLE_HIGH = np.array([2.4, 3.0, 0.2095, 3.5])


class Discretizer:
    def __init__(self, bins, low=CARTPOLE_LOW, high=CARTPOLE_HIGH):
        self.bins = bins
        # Solo los bordes interiores: digitize manda todo lo que está fuera del rango al primer
        # o último bin, que es justo el recorte de velocidades acordado en H0.
        self.edges = [np.linspace(lo, hi, bins + 1)[1:-1] for lo, hi in zip(low, high)]

    def __call__(self, obs):
        return tuple(int(np.digitize(o, e)) for o, e in zip(obs, self.edges))
