"""Entrena un agente con seed fija y deja results/<env>/<algo>/<tag>/seed<k>.csv + config.json.

Desde Project1/:
    python -m experiments.run --algo tabular --bins 6 10 20 --seeds 0 1 2 3 4 --jobs 4
    python -m experiments.run --algo dqn --seeds 0 1 2 3 4 --jobs 5
    python -m experiments.run --env LunarLander-v3 --algo dqn --vary lr 1e-4 1e-3 --steps 300000 --seeds 0 1 2 --jobs 6
"""
import argparse
import json
import platform
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
STEPS = {"CartPole-v1": 200_000, "LunarLander-v3": 500_000}
FINAL_EVAL_EPISODES = 100

TABULAR = {"alpha": 0.1, "gamma": 0.99, "eps_end": 0.05, "eps_decay_frac": 0.1}
# Defaults de DQN acordados en H0 (tabla del README).
DQN_DEFAULTS = {
    "CartPole-v1": {"hidden": 128, "lr": 1e-3, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 64,
                    "learning_starts": 1_000, "train_every": 1, "target_sync": 500,
                    "use_target": True, "eps_end": 0.05, "eps_decay_frac": 0.10},
    "LunarLander-v3": {"hidden": 256, "lr": 5e-4, "gamma": 0.99, "buffer_size": 100_000, "batch_size": 128,
                       "learning_starts": 10_000, "train_every": 4, "target_sync": 1_000,
                       "use_target": True, "eps_end": 0.05, "eps_decay_frac": 0.12},
}


def cpu_name():
    """Modelo de CPU para config.json (la orden pide reportar el hardware)."""
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()   # Windows / macOS


def make_agent(cfg, env, seed):
    eps_decay_steps = int(cfg["eps_decay_frac"] * cfg["steps"])
    if cfg["algo"] == "tabular":
        from agents.tabular import QLearning
        return QLearning(env.action_space.n, cfg["bins"], seed, eps_decay_steps=eps_decay_steps,
                         alpha=cfg["alpha"], gamma=cfg["gamma"], eps_end=cfg["eps_end"])
    if cfg["algo"] == "dqn":
        from agents.dqn import DQN
        return DQN(env.observation_space.shape[0], env.action_space.n, seed, eps_decay_steps,
                   hidden=cfg["hidden"], lr=cfg["lr"], gamma=cfg["gamma"],
                   buffer_size=cfg["buffer_size"], batch_size=cfg["batch_size"],
                   learning_starts=cfg["learning_starts"], train_every=cfg["train_every"],
                   target_sync=cfg["target_sync"], use_target=cfg["use_target"], eps_end=cfg["eps_end"])
    raise ValueError(f"algo desconocido: {cfg['algo']}")


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
        if step % eval_every == 0 or step == cfg["steps"]:
            # La última fila usa 100 episodios: ese es el número que se compara con el umbral.
            n = FINAL_EVAL_EPISODES if step == cfg["steps"] else 10
            mean, std = evaluate(agent, cfg["env"], seed, n_episodes=n)
            log.write(step=step, eval_mean=mean, eval_std=std,
                      train_return=np.mean(last10) if last10 else np.nan,
                      episodes=episodes, wall_s=round(time.perf_counter() - t0, 2))
    log.close()
    env.close()
    if cfg["algo"] == "dqn":
        torch.save(agent.q.state_dict(), out / f"seed{seed}.pt")   # para los videos (H9)
    else:
        np.save(out / f"seed{seed}_q.npy", agent.q)                # tabla Q entrenada
    print(f"{cfg['tag']} seed{seed}: final {mean:.1f} ± {std:.1f} en {time.perf_counter() - t0:.0f} s")


def _run(args):
    return run(*args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1", choices=list(STEPS))
    p.add_argument("--algo", default="tabular", choices=["tabular", "dqn"])
    p.add_argument("--steps", type=int, help="por defecto el presupuesto del entorno (README)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--bins", type=int, nargs="+", default=[10], help="solo tabular")
    p.add_argument("--vary", nargs="+", metavar="KEY VAL",
                   help="solo dqn: un hiperparámetro y sus valores, p. ej. --vary target_sync 250 4000")
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()

    if a.algo == "tabular":
        variants = [(f"bins{b}", {**TABULAR, "bins": b}) for b in a.bins]
    elif a.vary:
        key, *vals = a.vary
        base = DQN_DEFAULTS[a.env]
        if key not in base or not vals:
            p.error(f"--vary necesita una clave de {sorted(base)} y al menos un valor")
        # json.loads convierte "1e-4" -> 0.0001, "250" -> 250, "false" -> False
        variants = [(f"{key}{v}", {**base, key: json.loads(v)}) for v in vals]
    else:
        variants = [("default", DQN_DEFAULTS[a.env])]

    steps = a.steps or STEPS[a.env]
    suffix = "" if steps == STEPS[a.env] else f"_{steps // 1000}k"   # default (500k) ≠ default_300k
    jobs = [({"env": a.env, "algo": a.algo, "tag": tag + suffix, "steps": steps, **hp,
              "eval_every": EVAL_EVERY[a.env], "final_eval_episodes": FINAL_EVAL_EPISODES,
              "hardware": cpu_name()}, s)
            for (tag, hp), s in product(variants, a.seeds)]
    with Pool(a.jobs) as pool:
        pool.map(_run, jobs)
