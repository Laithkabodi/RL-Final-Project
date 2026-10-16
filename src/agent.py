"""Double DQN agent around the fixed QNetwork."""
import numpy as np
import torch
import torch.nn.functional as F

from .network import QNetwork


class DQNAgent:
    def __init__(self, state_dim, action_dim, cfg, seed):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.action_dim = action_dim
        self.q = QNetwork(state_dim, action_dim)
        self.q_target = QNetwork(state_dim, action_dim)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, fused=True)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def q_values(self, state):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.q(s).squeeze(0).numpy()

    def greedy_action(self, state):
        return int(np.argmax(self.q_values(state)))

    def worst_action(self, state):
        """Action with the lowest estimated Q-value (used for adversarial kicks)."""
        return int(np.argmin(self.q_values(state)))

    def epsilon_greedy(self, state, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.action_dim))
        return self.greedy_action(state)

    def update(self, batch):
        s, a, r, s2, term = batch
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg.double_dqn:
                # Double DQN: online network chooses action, target network computes its value
                a2 = self.q(s2).argmax(dim=1, keepdim=True)
                q_next = self.q_target(s2).gather(1, a2).squeeze(1)
            else:
                q_next = self.q_target(s2).max(dim=1).values
            target = r + self.cfg.gamma * (1.0 - term) * q_next
        loss = F.smooth_l1_loss(q_sa, target)  # Huber loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip, foreach=True)
        self.opt.step()
        self._soft_update()
        return float(loss.item())

    @torch.no_grad()
    def _soft_update(self):
        # Soft update of target network weights using Polyak averaging
        torch._foreach_lerp_(list(self.q_target.parameters()),
                             list(self.q.parameters()), self.cfg.tau)
