"""DQN con replay buffer y target network (Tier 1 y 2, H3). Mismo patrón que el lab de DQN."""
import copy

import numpy as np
import torch
from torch import nn


# --- Sección 1: replay buffer y red Q ------------------------------------------------------

class ReplayBuffer:
    """Buffer circular: al llenarse, cada transición nueva pisa la más vieja."""

    def __init__(self, capacity, state_dim, rng):
        self.N, self.c, self.rng = capacity, 0, rng   # c = transiciones agregadas en total
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, s, a, r, s2, done):
        i = self.c % self.N                           # mismo índice i en los cinco arreglos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, done
        self.c += 1

    def __len__(self):
        return min(self.c, self.N)

    def sample(self, n):
        idx = self.rng.integers(0, len(self), size=n)
        return tuple(torch.as_tensor(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.done))


class QNetwork(nn.Module):
    """s -> [Q(s, a) para cada acción]. Sin activación al final: Q puede ser negativo."""

    def __init__(self, state_dim, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)                            # (batch, n_actions)


# --- Sección 2: el agente ------------------------------------------------------------------

class DQN:
    """Mismo contrato que el tabular: act(obs, greedy) y update(obs, a, r, obs2, terminated, truncated)."""

    def __init__(self, state_dim, n_actions, seed, eps_decay_steps, hidden, lr, gamma,
                 buffer_size, batch_size, learning_starts, train_every, target_sync,
                 use_target=True, eps_start=1.0, eps_end=0.05):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.q = QNetwork(state_dim, n_actions, hidden)
        # Ablación sin target net: el target usa la misma red que se está entrenando (θ⁻ = θ).
        self.q_target = copy.deepcopy(self.q) if use_target else self.q
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, state_dim, self.rng)
        self.gamma, self.batch_size = gamma, batch_size
        self.learning_starts, self.train_every = learning_starts, train_every
        self.target_sync, self.use_target = target_sync, use_target
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
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))   # (1, n_actions)
        return int(q.argmax(1).item())

    def update(self, obs, action, reward, next_obs, terminated, truncated):
        # Solo terminated corta el bootstrap: truncated (límite de tiempo) no hace terminal al estado.
        self.buffer.add(obs, action, reward, next_obs, float(terminated))
        self.steps += 1
        if self.steps >= self.learning_starts and self.steps % self.train_every == 0:
            self.learn()
        if self.use_target and self.steps % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())             # θ⁻ ← θ

    def next_value(self, s2):
        """máx_a' Q(s', a'; θ⁻). Double DQN (H6) solo cambia este método."""
        return self.q_target(s2).max(1).values

    def learn(self):
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)              # Q(s, a; θ) de la acción tomada
        with torch.no_grad():                                               # el target no recibe gradiente
            y = r + self.gamma * (1 - done) * self.next_value(s2)
        loss = nn.functional.mse_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
