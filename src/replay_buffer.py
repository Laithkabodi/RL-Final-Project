"""Experience replay buffer for storing and sampling transitions."""
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity circular buffer backed by pre-allocated numpy arrays."""

    def __init__(self, capacity, state_dim, rng):
        self.capacity = capacity
        self.rng = rng
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # Store 1.0 only when the episode ended because the pole fell or cart went out of bounds.
        # Max-step timeouts store 0.0 so the Q-value can properly bootstrap beyond 1,000 steps.
        self.terminals = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, s, a, r, s2, terminal):
        self.states[self.idx] = s
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.next_states[self.idx] = s2
        self.terminals[self.idx] = float(terminal)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ix = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[ix]),
            torch.from_numpy(self.actions[ix]),
            torch.from_numpy(self.rewards[ix]),
            torch.from_numpy(self.next_states[ix]),
            torch.from_numpy(self.terminals[ix]),
        )

    def __len__(self):
        return self.size
