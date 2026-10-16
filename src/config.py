"""Shared hyperparameters across all training strategies.

All strategies use identical DQN settings and training budgets (200k steps).
Strategies only vary in how pole lengths are sampled and whether disturbance
kicks are introduced during training.
"""
from dataclasses import dataclass, asdict


# Target pole length bounds from the assignment test protocol (0.4m to 1.8m)
LENGTH_MIN = 0.4
LENGTH_MAX = 1.8
DEFAULT_LENGTH = 0.5

# Validation lengths for checkpoint evaluation.
# We pick lengths between test grid points and evaluate under normal dynamics
# so model selection remains independent of the test protocol.
VALIDATION_LENGTHS = (0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.78)


@dataclass
class DQNConfig:
    # -- budget -----------------------------------------------------------
    total_steps: int = 200_000        # environment steps per training run
    max_episode_steps: int = 1_000    # training episode cap (bootstrapped)
    # --- DQN --------------------------------------------------------
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 2_000
    train_freq: int = 1               # one gradient step every n env steps
    tau: float = 0.005                # Polyak factor for the target network
    grad_clip: float = 10.0
    double_dqn: bool = True
    # --- exploration (linear epsilon decay) ----------------------------
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_fraction: float = 0.3   # fraction of total_steps
    # -- checkpoint selection --------------------------------------------
    eval_every: int = 5_000
    eval_episodes_per_length: int = 2

    def to_dict(self):
        return asdict(self)
