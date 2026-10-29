"""Q-learning tabular sobre un discretizador (Tier 1, H2)."""
import numpy as np

from envs.discretizer import Discretizer


class QLearning:
    def __init__(self, n_actions, bins, seed, eps_decay_steps,
                 alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05):
        self.disc = Discretizer(bins)
        self.q = np.zeros((bins,) * 4 + (n_actions,))
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.steps = 0

    @property
    def eps(self):
        """ε lineal de eps_start a eps_end en eps_decay_steps pasos, luego constante."""
        frac = min(self.steps / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[self.disc(obs)]))

    def update(self, obs, action, reward, next_obs, terminated, truncated):
        s, s2 = self.disc(obs), self.disc(next_obs)
        # terminated: el polo cayó, no hay futuro. truncated (límite de 500 pasos) sí se
        # bootstrapea: el estado no es terminal, solo se cortó el episodio.
        target = reward if terminated else reward + self.gamma * self.q[s2].max()
        self.q[s + (action,)] += self.alpha * (target - self.q[s + (action,)])
        self.steps += 1
