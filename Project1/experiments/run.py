"""Entrena un agente con seed fija y deja results/<env>/<algo>/<tag>/seed<k>.csv + config.json.

Desde Project1/:
    python -m experiments.run --algo tabular --bins 6 10 20 --seeds 0 1 2 3 4 --jobs 4
"""
import argparse
import json
import time
from collections import deque
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.common import CsvLog, evaluate, set_seed

RESULTS = Path(__file__).resolve().parent.parent / "results"
EVAL_EVERY = {"CartPole-v1": 5_000, "LunarLander-v3": 10_000}
FINAL_EVAL_EPISODES = 100


def make_agent(cfg, env, seed):
    if cfg["algo"] == "tabular":
        from agents.tabular import QLearning
        return QLearning(env.action_space.n, cfg["bins"], seed,
                         eps_decay_steps=int(cfg["eps_decay_frac"] * cfg["steps"]),
                         alpha=cfg["alpha"], gamma=cfg["gamma"], eps_end=cfg["eps_end"])
    raise ValueError(f"algo desconocido: {cfg['algo']}")  # DQN se agrega aquí (H3)


def run(cfg, seed):
    torch.set_num_threads(1)
    set_seed(seed)
    out = RESULTS / cfg["env"] / cfg["algo"] / cfg["tag"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    env = gym.make(cfg["env"])
    agent = make_agent(cfg, env, seed)
    log = CsvLog(out / f"seed{seed}.csv")
    eval_every = EVAL_EVERY[cfg["env"]]
    last10, ep_return, episodes = deque(maxlen=10), 0.0, 0
    t0 = time.perf_counter()

    obs, _ = env.reset(seed=seed)
    for step in range(1, cfg["steps"] + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated, truncated)
        obs, ep_return = next_obs, ep_return + reward
        if terminated or truncated:
            last10.append(ep_return)
            episodes += 1
            obs, ep_return = env.reset()[0], 0.0
        if step % eval_every == 0:
            # La última fila usa 100 episodios: ese es el número que se compara con el umbral.
            n = FINAL_EVAL_EPISODES if step == cfg["steps"] else 10
            mean, std = evaluate(agent, cfg["env"], seed, n_episodes=n)
            log.write(step=step, eval_mean=mean, eval_std=std,
                      train_return=np.mean(last10) if last10 else np.nan,
                      episodes=episodes, wall_s=round(time.perf_counter() - t0, 2))
    log.close()
    env.close()
    print(f"{cfg['tag']} seed{seed}: final {mean:.1f} ± {std:.1f} en {time.perf_counter() - t0:.0f} s")


def _run(args):
    return run(*args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--algo", default="tabular")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--bins", type=int, nargs="+", default=[10])
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()

    jobs = [({"env": a.env, "algo": a.algo, "tag": f"bins{b}", "steps": a.steps, "bins": b,
              "alpha": 0.1, "gamma": 0.99, "eps_end": 0.05, "eps_decay_frac": 0.1,
              "eval_every": EVAL_EVERY[a.env], "final_eval_episodes": FINAL_EVAL_EPISODES}, s)
            for b, s in product(a.bins, a.seeds)]
    with Pool(a.jobs) as pool:
        pool.map(_run, jobs)
