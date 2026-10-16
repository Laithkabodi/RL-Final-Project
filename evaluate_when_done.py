"""Evaluates training runs in the background as soon as they complete.

Monitors results/logs/ and automatically evaluates finished model checkpoints.

Usage:
    python evaluate_when_done.py --workers 3 --expected 40
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

from src.evaluate import _job, ROOT

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--expected", type=int, default=40)
    a = p.parse_args()
    os.makedirs(os.path.join(ROOT, "results", "eval"), exist_ok=True)
    submitted, futures = set(), []
    with ProcessPoolExecutor(a.workers) as pool:
        while len(submitted) < a.expected:
            for log in glob.glob(os.path.join(ROOT, "results", "logs", "*.json")):
                tag = os.path.splitext(os.path.basename(log))[0]
                if tag not in submitted:
                    submitted.add(tag)
                    path = os.path.join(ROOT, "weights", "runs", tag + ".pth")
                    futures.append(pool.submit(_job, path))
            for f in [f for f in futures if f.done()]:
                print(f.result(), flush=True)
                futures.remove(f)
            time.sleep(20)
        for f in futures:
            print(f.result(), flush=True)
