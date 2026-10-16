"""Selects the best checkpoint for each strategy and runs the official test script.

Model selection is strictly based on the training-time validation score (evaluated
under normal dynamics on validation pole lengths separate from the test grid).
The selected weights are copied to weights/<strategy>.pth and tested using the
provided test_script.py. Results (bar plots and Excel files) are saved to
results/official/<strategy>/.

Usage:
    python select_and_test.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
STRATEGIES = ["baseline", "uniform", "adaptive", "adversarial"]


def select(strategy):
    best = None
    for path in glob.glob(os.path.join(ROOT, "results", "logs", f"{strategy}_seed*.json")):
        log = json.load(open(path))
        key = (log["best_val_score"], -log["seed"])  # ties -> lowest seed
        if best is None or key > best[0]:
            best = (key, log["seed"])
    seed = best[1]
    src = os.path.join(ROOT, "weights", "runs", f"{strategy}_seed{seed}.pth")
    shutil.copy(src, os.path.join(ROOT, "weights", f"{strategy}.pth"))
    return strategy, seed, best[0][0]


def official_test(strategy):
    out_dir = os.path.join(ROOT, "results", "official", strategy)
    os.makedirs(out_dir, exist_ok=True)
    # test_script.py loads "weights/<file>" relative to the cwd and writes its
    # outputs to the cwd, so run it from a folder containing a weights/ link.
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "weights", f"{strategy}.pth"),
                os.path.join(out_dir, "weights", f"{strategy}.pth"))
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with open(os.path.join(out_dir, "stdout.txt"), "w") as f:
        code = subprocess.call([sys.executable, os.path.join(ROOT, "test_script.py"),
                                f"{strategy}.pth"], cwd=out_dir, env=env,
                               stdout=f, stderr=subprocess.STDOUT)
    shutil.rmtree(os.path.join(out_dir, "weights"))
    return f"{strategy}: official test exit {code}"


if __name__ == "__main__":
    selection = {}
    for s in STRATEGIES:
        strategy, seed, score = select(s)
        selection[s] = {"seed": seed, "val_score": score}
        print(f"{s}: selected seed {seed} (validation score {score:.2f})")
    os.makedirs(os.path.join(ROOT, "results", "official"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "official", "selection.json"), "w") as f:
        json.dump(selection, f, indent=1)
    with ThreadPoolExecutor(4) as pool:
        for msg in pool.map(official_test, STRATEGIES):
            print(msg, flush=True)
