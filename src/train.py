"""Training script for running a single DQN agent with a chosen strategy and seed.

Usage:
    python -m src.train --strategy uniform --seed 0

Outputs:
    weights/runs/<strategy>_seed<seed>.pth   (best checkpoint based on validation)
    results/logs/<strategy>_seed<seed>.csv  (episode training logs)
    results/logs/<strategy>_seed<seed>.json (run config and validation records)
"""
import argparse
import json
import os
import time

# Keep BLAS single-threaded per process for faster execution when running parallel seeds
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gym
import numpy as np
import torch

from .agent import DQNAgent
from .config import DQNConfig, VALIDATION_LENGTHS, DEFAULT_LENGTH
from .replay_buffer import ReplayBuffer
from .strategies import build_strategy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_env(length, seed=None):
    env = gym.make("CartPole-v1")
    env.unwrapped.length = length
    state, _ = env.reset(seed=seed)
    return env, state


def validate(agent, lengths, cfg, seed):
    """Evaluates greedy rollouts on held-out validation pole lengths under normal dynamics.

    Returns a score combining the mean return with a small stability margin bonus
    (rewarding carts that stay closer to the center and upright) to cleanly break ties.
    """
    returns, margins = [], []
    for j, length in enumerate(lengths):
        for e in range(cfg.eval_episodes_per_length):
            env, s = make_env(length, seed=seed * 1000 + j * 10 + e)
            dev, ret = 0.0, 0
            for _ in range(cfg.max_episode_steps):
                s, r, term, _, _ = env.step(agent.greedy_action(s))
                ret += r
                dev += 0.5 * (abs(s[0]) / 2.4 + abs(s[2]) / 0.2095)
                if term:
                    break
            returns.append(ret)
            margins.append(1.0 - min(dev / max(ret, 1), 1.0))
            env.close()
    mean_ret = float(np.mean(returns))
    return mean_ret + 0.999 * float(np.mean(margins)), mean_ret


def train(strategy, seed, cfg=None, verbose=True, tag=None):
    cfg = cfg or DQNConfig()
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    sampler, disturbance = build_strategy(strategy, rng)
    val_lengths = (DEFAULT_LENGTH,) if strategy == "baseline" else VALIDATION_LENGTHS

    env, state = make_env(sampler.sample(), seed=seed)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    agent = DQNAgent(state_dim, action_dim, cfg, seed)
    buffer = ReplayBuffer(cfg.buffer_size, state_dim, rng)

    os.makedirs(os.path.join(ROOT, "weights", "runs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    tag = tag or f"{strategy}_seed{seed}"
    ckpt_path = os.path.join(ROOT, "weights", "runs", tag + ".pth")

    log_rows, val_hist = [], []
    best_score = -np.inf
    length = env.unwrapped.length
    ep_ret, ep_len, ep_disturbed, episode = 0.0, 0, 0, 0
    disturbance.reset()
    t0 = time.time()

    for step in range(1, cfg.total_steps + 1):
        frac = min(step / (cfg.eps_decay_fraction * cfg.total_steps), 1.0)
        eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        action = agent.epsilon_greedy(state, eps)
        action, disturbed = disturbance.act(agent, state, action)

        next_state, reward, terminated, _, _ = env.step(action)
        buffer.add(state, action, reward, next_state, terminated)
        state = next_state
        ep_ret += reward
        ep_len += 1
        ep_disturbed += int(disturbed)

        if step > cfg.learning_starts and step % cfg.train_freq == 0:
            agent.update(buffer.sample(cfg.batch_size))

        # End of episode: pole fell / cart out of bounds, or hit training step cap
        if terminated or ep_len >= cfg.max_episode_steps:
            sampler.update(length, ep_ret, cfg.max_episode_steps)
            log_rows.append((episode, step, round(length, 4), ep_ret, ep_disturbed, round(eps, 4)))
            episode += 1
            env.close()
            env, state = make_env(sampler.sample())
            length = env.unwrapped.length
            ep_ret, ep_len, ep_disturbed = 0.0, 0, 0
            disturbance.reset()

        if step % cfg.eval_every == 0:
            score, mean_ret = validate(agent, val_lengths, cfg, seed)
            improved = score >= best_score  # On ties, prefer the later, more trained checkpoint
            if improved:
                best_score = score
                torch.save(agent.q.state_dict(), ckpt_path)
            val_hist.append({"step": step, "val_return": mean_ret,
                             "val_score": round(score, 4), "saved": improved,
                             "sampler": sampler.state()})
            if verbose:
                recent = np.mean([r[3] for r in log_rows[-20:]]) if log_rows else 0
                print(f"[{tag}] step {step:>7} | eps {eps:.2f} | train(last20) {recent:7.1f}"
                      f" | val {mean_ret:7.1f} {'*' if improved else ''} | {time.time() - t0:5.0f}s",
                      flush=True)

    env.close()
    with open(os.path.join(ROOT, "results", "logs", tag + ".csv"), "w") as f:
        f.write("episode,step,length,return,disturbed_steps,epsilon\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in log_rows)
    with open(os.path.join(ROOT, "results", "logs", tag + ".json"), "w") as f:
        json.dump({"strategy": strategy, "seed": seed, "config": cfg.to_dict(),
                   "best_val_score": best_score, "train_time_s": time.time() - t0,
                   "validation": val_hist}, f, indent=1)
    return ckpt_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, choices=["baseline", "uniform", "adaptive", "adversarial"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=None)
    a = p.parse_args()
    cfg = DQNConfig()
    if a.total_steps:
        cfg.total_steps = a.total_steps
    train(a.strategy, a.seed, cfg)
