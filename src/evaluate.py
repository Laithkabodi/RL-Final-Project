"""Batch evaluation of trained models using the assignment test protocol.

Runs the test_pole_length rollout across 30 pole lengths (10 episodes each)
covering the nominal, wind, and increasing-force phases. Initial states are
consistently seeded so comparisons between seeds and strategies are fair.

Usage:
    python -m src.evaluate
    python -m src.evaluate --workers 8
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gym
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from test_script import test_pole_length  # noqa: E402  (official rollout)
from src.network import QNetwork  # noqa: E402

POLE_LENGTHS = np.linspace(0.4, 1.8, 30)
EPISODES_PER_LENGTH = 10


def evaluate_model(weight_path, eval_seed=12345):
    torch.set_num_threads(1)
    model = QNetwork(4, 2)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()
    rows = []
    for i, length in enumerate(POLE_LENGTHS):
        for k in range(EPISODES_PER_LENGTH):
            env = gym.make("CartPole-v1")
            env.unwrapped.length = length
            # Pre-seed the environment so all models face identical starting conditions
            env.reset(seed=eval_seed + 100 * i + k)
            with torch.no_grad():
                score = test_pole_length(env, model)
            env.close()
            rows.append((round(float(length), 4), k, score))
    return rows


class NumpyQ:
    """Fast NumPy forward pass using the trained weights (avoids PyTorch tensor dispatch overhead)."""

    def __init__(self, weight_path):
        sd = torch.load(weight_path, weights_only=True)
        self.w = [sd[f"fc{i}.weight"].numpy().T.copy() for i in (1, 2, 3)]
        self.b = [sd[f"fc{i}.bias"].numpy().copy() for i in (1, 2, 3)]

    def action(self, s):
        h = np.maximum(s @ self.w[0] + self.b[0], 0.0)
        h = np.maximum(h @ self.w[1] + self.b[1], 0.0)
        return int(np.argmax(h @ self.w[2] + self.b[2]))


def fast_test_pole_length(env, qnet):
    """Equivalent rollout to test_script.py using NumPy forward pass for speed.
    
    Applies the exact same perturbation schedule (wind force 75 from step 500,
    then force_mag = 25 + 0.01 * t past step 1000).
    """
    wind = 25
    state = np.asarray(env.reset()[0], dtype=np.float32)
    done = False
    total_reward = 0
    while not done:
        action = qnet.action(state)
        next_state, reward, done, _, __ = env.step(action)
        state = np.asarray(next_state, dtype=np.float32)
        total_reward += reward
        if total_reward >= 500 and total_reward <= 1000:
            if total_reward % wind == 0:
                env.force_mag = 75
        if total_reward > 1000:
            env.force_mag = 25 + (0.01 * total_reward)
    return total_reward


def fast_evaluate_model(weight_path, eval_seed=12345):
    qnet = NumpyQ(weight_path)
    rows = []
    for i, length in enumerate(POLE_LENGTHS):
        for k in range(EPISODES_PER_LENGTH):
            env = gym.make("CartPole-v1").unwrapped
            env.length = length
            env.reset(seed=eval_seed + 100 * i + k)  # same seeding as above
            rows.append((round(float(length), 4), k, fast_test_pole_length(env, qnet)))
            env.close()
    return rows


def _job(path, official=False):
    tag = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(ROOT, "results", "eval", tag + ".csv")
    if os.path.exists(out):
        return f"{tag}: skipped"
    t0 = time.time()
    rows = evaluate_model(path) if official else fast_evaluate_model(path)
    with open(out, "w") as f:
        f.write("length,episode,score\n")
        f.writelines(f"{l},{k},{s}\n" for l, k, s in rows)
    mean = np.mean([r[2] for r in rows])
    return f"{tag}: mean {mean:8.1f}  ({time.time() - t0:.0f}s)"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="*.pth")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--official", action="store_true",
                   help="use the official torch rollout instead of the fast replica")
    a = p.parse_args()
    os.makedirs(os.path.join(ROOT, "results", "eval"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(ROOT, "weights", "runs", a.pattern)))
    with ProcessPoolExecutor(a.workers) as pool:
        for msg in pool.map(_job, paths, [a.official] * len(paths)):
            print(msg, flush=True)
