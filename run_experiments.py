"""Parallel runner to train multiple strategies and random seeds.

Usage:
    python run_experiments.py --seeds 0-9 --workers 8
    python run_experiments.py --strategies uniform adaptive --seeds 0-2

Skips runs that already have completed logs, making it safe to resume if interrupted.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
ALL = ["baseline", "uniform", "adaptive", "adversarial"]


def parse_seeds(text):
    if "-" in text:
        lo, hi = map(int, text.split("-"))
        return list(range(lo, hi + 1))
    return [int(s) for s in text.split(",")]


def run(job):
    strategy, seed = job
    tag = f"{strategy}_seed{seed}"
    if os.path.exists(os.path.join(ROOT, "results", "logs", tag + ".json")):
        return f"{tag}: skipped (exists)"
    os.makedirs(os.path.join(ROOT, "results", "stdout"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "stdout", tag + ".txt"), "w") as out:
        code = subprocess.call([sys.executable, "-W", "ignore", "-m", "src.train",
                                "--strategy", strategy, "--seed", str(seed)],
                               cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    return f"{tag}: exit {code}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    # Interleave strategies so that partial results cover all strategies.
    jobs = [(s, seed) for seed in parse_seeds(a.seeds) for s in a.strategies]
    with ThreadPoolExecutor(a.workers) as pool:
        for msg in pool.map(run, jobs):
            print(msg, flush=True)
