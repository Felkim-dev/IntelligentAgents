"""Graba un episodio greedy de un DQN ya entrenado como GIF (H9).

Desde Project1/:
    python -m experiments.gif --env CartPole-v1 --seed 0
    python -m experiments.gif --env LunarLander-v3 --seed 0
"""
import argparse
import json

import gymnasium as gym
import imageio.v3 as iio
import torch

from agents.dqn import QNetwork
from experiments.run import RESULTS

FIGURES = RESULTS.parent / "figures"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="default")
    a = p.parse_args()

    # La red se reconstruye con el mismo `hidden` con que se entrenó; si no, los pesos no encajan.
    run_dir = RESULTS / a.env / "dqn" / a.tag
    cfg = json.loads((run_dir / "config.json").read_text())
    env = gym.make(a.env, render_mode="rgb_array")
    q = QNetwork(env.observation_space.shape[0], env.action_space.n, cfg["hidden"])
    q.load_state_dict(torch.load(run_dir / f"seed{a.seed}.pt"))

    # Un episodio con la acción de mayor Q (sin ε): es la política aprendida, no la de exploración.
    obs, _ = env.reset(seed=a.seed)
    frames, ret, done = [env.render()], 0.0, False
    while not done:
        with torch.no_grad():
            action = q(torch.as_tensor(obs, dtype=torch.float32)).argmax().item()
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        ret += reward
        done = terminated or truncated
    env.close()

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / f"{a.env}_{a.tag}_seed{a.seed}.gif"
    iio.imwrite(out, frames, duration=1000 / env.metadata["render_fps"], loop=0)   # duration en ms por frame
    print(f"{out.name}: {len(frames)} frames, retorno {ret:.1f}")
