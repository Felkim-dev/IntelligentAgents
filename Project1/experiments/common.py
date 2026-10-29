"""Piezas comunes a todos los experimentos: seeds, evaluación greedy y CSV.

Contrato que cumple todo agente (tabular o DQN):
    agent.act(obs, greedy: bool) -> acción
    agent.update(obs, action, reward, next_obs, terminated, truncated) -> None
"""
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

CSV_COLUMNS = ["step", "eval_mean", "eval_std", "train_return", "episodes", "wall_s"]
EVAL_SEED_OFFSET = 1000  # el entorno de evaluación nunca comparte seed con el de entrenamiento


def set_seed(seed):
    """Fija todas las fuentes de azar del proceso (el entorno se fija aparte, en reset)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(agent, env_id, seed, n_episodes=10):
    """Corre n_episodes con ε = 0 en un entorno propio. Devuelve (media, desviación) del retorno."""
    env = gym.make(env_id)
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + EVAL_SEED_OFFSET + i)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, greedy=True))
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


class CsvLog:
    """Un archivo por corrida: results/<env>/<algo>/<tag>/seed<k>.csv"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()

    def write(self, **row):
        self.writer.writerow(row)
        self.file.flush()  # si la corrida muere a medianoche, lo escrito hasta ahí se conserva

    def close(self):
        self.file.close()
