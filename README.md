# Assignment 3 — Group 9: CartPole DQN

This repository contains our code, trained models, and results for Assignment 3. We trained a standard neural network (4-128-128-2) using Deep Q-Learning (DQN) to balance CartPole-v1 across pole lengths from 0.4 to 1.8, and to survive the test wind and force perturbations.

**Research question:** How does the way the agent experiences variation during training (which pole lengths it sees, and whether it has to recover from disturbances) affect generalisation across pole lengths and robustness to force perturbations it never saw during training? The full write-up is in `report/main.tex`.

## What We Tested

We compared four different training strategies. All strategies used the exact same neural network, hyperparameters, and budget of 200,000 training steps:

- **Baseline:** Trained only on the default pole length of 0.5.
- **Uniform (Domain Randomization):** Trained on random pole lengths chosen uniformly between 0.4 and 1.8 each episode.
- **Adaptive Curriculum:** Binned the pole lengths and sampled lower-performing lengths more often to focus practice where needed.
- **Adversarial:** Trained on random lengths, but periodically forced 2 to 5 steps of the worst action to practice recovering from severe tilts.

## Key Results

Every strategy was trained across 10 random seeds and evaluated on 30 different pole lengths (10 episodes per length):

- **Training on many pole lengths is what matters:** Uniform training scored highest on average (14,618 steps), but its advantage over the adaptive and adversarial strategies was not statistically significant. All three clearly beat the baseline. Our best submitted model (`weights/uniform.pth`) achieved an average score of 35,483 steps on the official test script.
- **Training on one length fails:** The Baseline model averaged only 4,213 steps and failed within the first 500 steps in 77% of test episodes, getting worse the further the pole length was from 0.5. Six of its ten seeds never learned to balance reliably even at 0.5.
- **Adaptive curriculum performed similarly to uniform:** It averaged 12,186 steps. During training, all pole lengths turned out to be similarly difficult, so the curriculum's priorities stayed almost flat and it ended up sampling nearly uniformly.
- **Adversarial kicks did not stop the wind:** Adversarial training averaged 11,442 steps. Learning to recover from bad actions under normal force (10) did not prepare the cart for the massive force increase during the wind phase (force 75).

## Submitted Models

The best checkpoint for each strategy (chosen based on validation during training):

- `weights/uniform.pth` (Uniform, seed 4) — Top score: **35,483** steps
- `weights/adaptive.pth` (Adaptive, seed 9) — Score: 19,629 steps
- `weights/adversarial.pth` (Adversarial, seed 1) — Score: 14,643 steps
- `weights/baseline.pth` (Baseline, seed 0) — Score: 9,156 steps

Official bar plots and Excel sheets for these models are stored in `results/official/`.

## How to Run

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Test a Trained Model
Run the official test script with any model weights:

```bash
# Test our best model
python test_script.py uniform.pth

# Test other models
python test_script.py adaptive.pth
python test_script.py adversarial.pth
python test_script.py baseline.pth
```
This produces `bar_plot.png` and `experiment_results.xlsx`. Note that the script calls `plt.show()`, so a plot window opens at the end; close it to let the script finish.

### 3. Re-train or Re-evaluate (Optional)
```bash
# Train all models (40 runs)
python run_experiments.py --seeds 0-9 --workers 8

# Evaluate all runs on the test grid
python -m src.evaluate --workers 8

# Generate statistics and figures
python -m src.analysis

# Select the best models and run the official test script
python select_and_test.py
```

## Project Files

- `test_script.py` — The assignment test script.
- `weights/` — Saved model checkpoints.
- `src/`
  - `network.py` — The Q-network architecture.
  - `agent.py` — Double DQN agent.
  - `strategies.py` — Pole length samplers and disturbance model.
  - `train.py` — Training loop.
  - `evaluate.py` — Batch test evaluator. For speed it uses a NumPy copy of the official test loop (same wind and force schedule). On identical starting states it gives exactly the same score as the official script in 97–99% of episodes, and the averages agree within 1%. The submitted models were also tested with the unmodified `test_script.py`.
  - `analysis.py` — Statistical tests and plot generation.
- `results/` — Training logs, test scores, and official test outputs.
- `figures/` — Generated plots and figures.
