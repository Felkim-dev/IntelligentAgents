"""Sesgo de Q (H6): máx_a Q(s, a) predicho vs retorno descontado real desde s, con la red final de cada seed.

Desde Project1/:
    python -m experiments.overestimation results/LunarLander-v3/dqn/default results/LunarLander-v3/dqn/doubletrue
Sesgo > 0 = la red sobreestima. Episodios greedy con las seeds de evaluación (seed + 1000 + i).
"""
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agents.dqn import QNetwork
from experiments.common import EVAL_SEED_OFFSET


def bias(run_dir, seed, n_episodes=10):
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    env = gym.make(cfg["env"])
    q = QNetwork(env.observation_space.shape[0], env.action_space.n, cfg["hidden"])
    q.load_state_dict(torch.load(Path(run_dir) / f"seed{seed}.pt"))
    diffs = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + EVAL_SEED_OFFSET + i)
        preds, rewards, done = [], [], False
        while not done:
            with torch.no_grad():
                qs = q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
            preds.append(qs.max().item())
            obs, r, terminated, truncated, _ = env.step(int(qs.argmax()))
            rewards.append(r)
            done = terminated or truncated
        g, returns = 0.0, []
        for r in reversed(rewards):                  # G_t = r_t + γ G_{t+1}
            g = r + cfg["gamma"] * g
            returns.append(g)
        # ponytail: si el episodio se trunca, G subestima el valor real de los últimos estados; en LunarLander
        # el aterrizaje termina el episodio mucho antes del límite, así que el efecto es chico.
        diffs.extend(np.array(preds) - np.array(returns[::-1]))
    env.close()
    return float(np.mean(diffs))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        seeds = sorted(int(f.stem[4:]) for f in Path(d).glob("seed*.pt"))
        b = [bias(d, s) for s in seeds]
        print(f"{Path(d).name}: sesgo de Q por seed {np.round(b, 2).tolist()} -> {np.mean(b):.2f} ± {np.std(b, ddof=1):.2f}")
