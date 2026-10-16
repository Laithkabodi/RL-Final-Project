"""Training strategies for CartPole generalization and robustness.

Each strategy combines:
  1. A pole-length sampler that selects the pole length for each training episode.
  2. An optional disturbance model that can inject action perturbations.

Strategies compared:
  - baseline    : Fixed length at 0.5m (standard Gym default)
  - uniform     : Uniform domain randomization across [0.4m, 1.8m]
  - adaptive    : Performance-based curriculum prioritizing struggling pole lengths
  - adversarial : Uniform domain randomization with occasional worst-action kicks
"""
import numpy as np

from .config import LENGTH_MIN, LENGTH_MAX, DEFAULT_LENGTH


# ============================================================ length samplers
class FixedLength:
    """Always samples the default fixed pole length (baseline)."""

    def __init__(self, rng, length=DEFAULT_LENGTH):
        self.length = length

    def sample(self):
        return self.length

    def update(self, length, episode_return, max_return):
        pass

    def state(self):
        return {}


class UniformLength:
    """Samples pole lengths uniformly at random (standard domain randomization)."""

    def __init__(self, rng, low=LENGTH_MIN, high=LENGTH_MAX):
        self.rng, self.low, self.high = rng, low, high

    def sample(self):
        return float(self.rng.uniform(self.low, self.high))

    def update(self, length, episode_return, max_return):
        pass

    def state(self):
        return {}


class AdaptiveCurriculum:
    """Adaptive curriculum sampler prioritizing pole lengths where the agent struggles.

    Divides the length range into discrete bins. Each bin tracks an exponential
    moving average of the return deficit (1 - R / R_max). Bins with lower returns
    receive higher sampling probabilities, while a minimum uniform mix prevents
    the agent from forgetting already-mastered lengths.
    """

    def __init__(self, rng, low=LENGTH_MIN, high=LENGTH_MAX, n_bins=14,
                 ema=0.3, beta=0.5, uniform_mix=0.2, eps=0.05):
        self.rng = rng
        self.edges = np.linspace(low, high, n_bins + 1)
        self.n_bins = n_bins
        self.deficit = np.ones(n_bins)
        self.counts = np.zeros(n_bins, dtype=int)
        self.ema, self.beta, self.uniform_mix, self.eps = ema, beta, uniform_mix, eps

    def probabilities(self):
        prio = (self.deficit + self.eps) ** (1.0 / self.beta)
        prio = prio / prio.sum()
        return self.uniform_mix / self.n_bins + (1.0 - self.uniform_mix) * prio

    def sample(self):
        i = self.rng.choice(self.n_bins, p=self.probabilities())
        return float(self.rng.uniform(self.edges[i], self.edges[i + 1]))

    def update(self, length, episode_return, max_return):
        i = min(np.searchsorted(self.edges, length, side="right") - 1, self.n_bins - 1)
        d = 1.0 - min(episode_return, max_return) / max_return
        self.deficit[i] = (1 - self.ema) * self.deficit[i] + self.ema * d
        self.counts[i] += 1

    def state(self):
        return {"deficit": self.deficit.round(4).tolist(),
                "counts": self.counts.tolist(),
                "probs": self.probabilities().round(4).tolist()}


# ========================================================= disturbance models
class NoDisturbance:
    def reset(self):
        pass

    def act(self, agent, state, action):
        return action, False


class SelfAdversarialBursts:
    """Action-level disturbance model using worst-action bursts.

    Instead of altering environment physics, this model uses the agent's own
    Q-network to deliver destabilizing kicks. At each step with probability
    `p_start`, a burst of 2-5 consecutive steps triggers where the agent takes
    its worst predicted action (lowest Q-value). This knocks the pole off-balance
    and trains the network to recover from difficult, near-failure states.
    Only activates after an initial warmup period once Q-estimates become meaningful.
    """

    def __init__(self, rng, p_start=0.01, k_min=2, k_max=5, start_step=10_000):
        self.rng = rng
        self.p_start, self.k_min, self.k_max = p_start, k_min, k_max
        self.start_step = start_step
        self.remaining = 0
        self.global_step = 0

    def reset(self):
        self.remaining = 0

    def act(self, agent, state, action):
        self.global_step += 1
        if self.global_step < self.start_step:
            return action, False
        if self.remaining == 0 and self.rng.random() < self.p_start:
            self.remaining = int(self.rng.integers(self.k_min, self.k_max + 1))
        if self.remaining > 0:
            self.remaining -= 1
            return agent.worst_action(state), True
        return action, False


# ================================================================== registry
STRATEGIES = {
    "baseline":    (FixedLength, NoDisturbance),
    "uniform":     (UniformLength, NoDisturbance),
    "adaptive":    (AdaptiveCurriculum, NoDisturbance),
    "adversarial": (UniformLength, SelfAdversarialBursts),
}


def build_strategy(name, rng):
    sampler_cls, disturbance_cls = STRATEGIES[name]
    disturbance = disturbance_cls(rng) if disturbance_cls is SelfAdversarialBursts else disturbance_cls()
    return sampler_cls(rng), disturbance
