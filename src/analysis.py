"""Generates summary statistics, significance tests, and figures for the assignment results.

Usage:
    python -m src.analysis
"""
import glob
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

KSTEPS = FuncFormatter(lambda x, _: f"{x / 1000:.0f}k")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

ORDER = ["baseline", "uniform", "adaptive", "adversarial"]
LABEL = {"baseline": "B: fixed length", "uniform": "S1: uniform DR",
         "adaptive": "S2: adaptive curriculum", "adversarial": "S3: self-adversarial"}
SHORT = {"baseline": "B", "uniform": "S1", "adaptive": "S2", "adversarial": "S3"}
COLOR = {"baseline": "#8a8985", "uniform": "#2a78d6", "adaptive": "#eb6834",
         "adversarial": "#1baf7a"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "savefig.bbox": "tight", "savefig.dpi": 300,
})


# ----------------------------------------------------------------- loading
def load_eval():
    frames = []
    for path in glob.glob(os.path.join(RES, "eval", "*.csv")):
        tag = os.path.splitext(os.path.basename(path))[0]
        strategy, seed = tag.rsplit("_seed", 1)
        if strategy not in ORDER:
            continue
        df = pd.read_csv(path)
        df["strategy"], df["seed"] = strategy, int(seed)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Phase in which the episode ended (see test_script.py):
    # 1: nominal (<=500 steps), 2: wind / force 75 (501-1000), 3: increasing force.
    df["phase"] = np.select([df.score <= 500, df.score <= 1000], [1, 2], 3)
    return df


def per_model(df):
    g = df.groupby(["strategy", "seed"])
    out = g.apply(lambda d: pd.Series({
        "overall": d.groupby("length").score.mean().mean(),
        "median_episode": d.score.median(),
        "fail_phase1": (d.phase == 1).mean(),
        "fail_phase2": (d.phase == 2).mean(),
        "reach_phase3": (d.phase == 3).mean(),
        # Wind survival rate among episodes that successfully reached the wind phase
        "wind_survival": ((d.score > 1000).sum() / (d.score > 500).sum()
                          if (d.score > 500).sum() >= 10 else np.nan),
        # Average steps survived in phase 3 (increasing force)
        "phase3_steps": (d.score[d.score > 1000] - 1000).mean() if (d.score > 1000).any() else 0.0,
    }), include_groups=False).reset_index()
    # Validation score recorded during training
    out["val_score"] = [json.load(open(os.path.join(RES, "logs", f"{s}_seed{k}.json")))["best_val_score"]
                        for s, k in zip(out.strategy, out.seed)]
    return out


# ------------------------------------------------------------------- stats
def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def statistics(models, df):
    res = {"per_strategy": {}, "pairwise": [], "paired_by_length": []}
    groups = {s: models[models.strategy == s].overall.values for s in ORDER}
    for s in ORDER:
        x = groups[s]
        ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
        res["per_strategy"][s] = {
            "n_seeds": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
            "ci95": [float(ci[0]), float(ci[1])],
            "fail_phase1": float(models[models.strategy == s].fail_phase1.mean()),
            "fail_phase2": float(models[models.strategy == s].fail_phase2.mean()),
            "reach_phase3": float(models[models.strategy == s].reach_phase3.mean()),
        }
    h, p = stats.kruskal(*[groups[s] for s in ORDER])
    res["kruskal_wallis"] = {"H": float(h), "p": float(p)}

    pairs = list(itertools.combinations(ORDER, 2))
    raw = []
    for a, b in pairs:
        u, p_u = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        t, p_t = stats.ttest_ind(groups[a], groups[b], equal_var=False)
        raw.append(p_u)
        res["pairwise"].append({"a": a, "b": b, "U": float(u), "p_mwu": float(p_u),
                                "welch_t": float(t), "p_welch": float(p_t),
                                "cliffs_delta": float(cliffs_delta(groups[a], groups[b])),
                                "mean_diff": float(groups[a].mean() - groups[b].mean())})
    for r, padj in zip(res["pairwise"], holm(raw)):
        r["p_mwu_holm"] = float(padj)

    # Secondary analysis: paired over the 30 pole lengths (seed-averaged),
    # using common random numbers (identical initial states per episode).
    by_len = df.groupby(["strategy", "length"]).score.mean().unstack(0)
    raw = []
    for a, b in pairs:
        w, p = stats.wilcoxon(by_len[a], by_len[b])
        raw.append(p)
        res["paired_by_length"].append({"a": a, "b": b, "W": float(w), "p": float(p),
                                        "a_better_on": int((by_len[a] > by_len[b]).sum())})
    for r, padj in zip(res["paired_by_length"], holm(raw)):
        r["p_holm"] = float(padj)

    # Comparison: fixed baseline (B) vs. diverse training strategies (S1-S3)
    b = models[models.strategy == "baseline"]
    d = models[models.strategy != "baseline"]
    res["planned_B_vs_diversity"] = {}
    for metric in ("overall", "fail_phase1"):
        u, p = stats.mannwhitneyu(b[metric], d[metric], alternative="two-sided")
        res["planned_B_vs_diversity"][metric] = {"U": float(u), "p": float(p),
                                                 "cliffs_delta": float(cliffs_delta(b[metric], d[metric]))}
    raw, rows = [], []
    for s in ORDER[1:]:
        x = models[models.strategy == s].fail_phase1
        u, p = stats.mannwhitneyu(b.fail_phase1, x, alternative="two-sided")
        raw.append(p)
        rows.append({"b": s, "U": float(u), "p": float(p), "cliffs_delta": float(cliffs_delta(b.fail_phase1, x))})
    for r, padj in zip(rows, holm(raw)):
        r["p_holm"] = float(padj)
    res["fail_phase1_B_vs"] = rows

    # Robustness metrics (per seed) and their group comparison.
    for metric in ("wind_survival", "phase3_steps", "fail_phase1"):
        g = {s: models[models.strategy == s][metric].dropna().values for s in ORDER}
        h, p = stats.kruskal(*[g[s] for s in ORDER])
        h3, p3 = stats.kruskal(*[g[s] for s in ORDER[1:]])
        res[metric] = {"n": {s: int(len(g[s])) for s in ORDER},
                       "min": {s: float(g[s].min()) for s in ORDER},
                       "max": {s: float(g[s].max()) for s in ORDER},
                       "mean": {s: float(g[s].mean()) for s in ORDER},
                       "median": {s: float(np.median(g[s])) for s in ORDER},
                       "kw_all": [float(h), float(p)], "kw_S1S2S3": [float(h3), float(p3)]}
    # Reliability: is the spread over seeds different? (Brown-Forsythe)
    w, p = stats.levene(*[groups[s] for s in ORDER[1:]], center="median")
    res["brown_forsythe_S1S2S3"] = {"W": float(w), "p": float(p)}
    res["pairwise_variance"] = []
    for a, b in itertools.combinations(ORDER[1:], 2):
        w, p = stats.levene(groups[a], groups[b], center="median")
        res["pairwise_variance"].append({"a": a, "b": b, "W": float(w), "p": float(p)})
    # Does the nominal-dynamics validation score predict test performance?
    rho, p = stats.spearmanr(models.val_score, models.overall)
    res["val_vs_test_spearman"] = {"rho": float(rho), "p": float(p)}
    return res


def write_stats(res):
    with open(os.path.join(RES, "stats.json"), "w") as f:
        json.dump(res, f, indent=1)
    lines = ["Overall score per agent (mean over 30 lengths x 10 episodes)", ""]
    lines.append(f"{'strategy':<12}{'n':>3}{'mean':>9}{'std':>9}{'median':>9}"
                 f"{'95% CI':>20}{'P1fail':>8}{'P2fail':>8}{'P3':>7}")
    for s, r in res["per_strategy"].items():
        lines.append(f"{s:<12}{r['n_seeds']:>3}{r['mean']:>9.0f}{r['std']:>9.0f}{r['median']:>9.0f}"
                     f"{'[%.0f, %.0f]' % tuple(r['ci95']):>20}{r['fail_phase1']:>8.2f}"
                     f"{r['fail_phase2']:>8.2f}{r['reach_phase3']:>7.2f}")
    kw = res["kruskal_wallis"]
    lines += ["", f"Kruskal-Wallis: H = {kw['H']:.2f}, p = {kw['p']:.4g}", "",
              "Pairwise (per-seed scores): Mann-Whitney U (Holm), Welch t, Cliff's delta"]
    for r in res["pairwise"]:
        lines.append(f"  {r['a']:>11} vs {r['b']:<11} U={r['U']:5.1f} p={r['p_mwu']:.4f} "
                     f"p_holm={r['p_mwu_holm']:.4f} | t={r['welch_t']:6.2f} p={r['p_welch']:.4f} "
                     f"| delta={r['cliffs_delta']:+.2f} | diff={r['mean_diff']:+.0f}")
    lines += ["", "Paired over pole lengths (seed-averaged): Wilcoxon signed-rank (Holm)"]
    for r in res["paired_by_length"]:
        lines.append(f"  {r['a']:>11} vs {r['b']:<11} W={r['W']:6.1f} p={r['p']:.4g} "
                     f"p_holm={r['p_holm']:.4g} | {r['a']} better on {r['a_better_on']}/30")
    lines += ["", "Planned contrast B vs pooled S1-S3 (Mann-Whitney U)"]
    for metric, r in res["planned_B_vs_diversity"].items():
        lines.append(f"  {metric:<12} U={r['U']:.0f} p={r['p']:.3g} delta={r['cliffs_delta']:+.2f}")
    lines.append("Phase-1 failure rate, B vs each strategy (Mann-Whitney U, Holm)")
    for r in res["fail_phase1_B_vs"]:
        lines.append(f"  B vs {r['b']:<11} U={r['U']:.0f} p={r['p']:.3g} p_holm={r['p_holm']:.3g} delta={r['cliffs_delta']:+.2f}")
    lines += ["", "Robustness metrics (per-seed means / medians; Kruskal-Wallis all 4 and S1-S3)"]
    for metric in ("wind_survival", "phase3_steps", "fail_phase1"):
        r = res[metric]
        lines.append(f"  {metric:<14}" + " ".join(f"{SHORT[s]}={r['mean'][s]:.3g}/{r['median'][s]:.3g}" for s in ORDER)
                     + f" | KW all p={r['kw_all'][1]:.4f} | KW S1-S3 p={r['kw_S1S2S3'][1]:.4f}")
    bf = res["brown_forsythe_S1S2S3"]
    lines += ["", f"Brown-Forsythe (seed variance, S1-S3): W={bf['W']:.2f} p={bf['p']:.4f}"]
    for r in res["pairwise_variance"]:
        lines.append(f"  {r['a']} vs {r['b']}: W={r['W']:.2f} p={r['p']:.4f}")
    sp = res["val_vs_test_spearman"]
    lines.append(f"Spearman(validation score, test score), n=40: rho={sp['rho']:.2f} p={sp['p']:.4f}")
    text = "\n".join(lines)
    with open(os.path.join(RES, "stats.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


# ----------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def fig_per_length(df):
    """Mean score per pole length; band = 95% CI over seeds."""
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    per_seed = df.groupby(["strategy", "seed", "length"]).score.mean().reset_index()
    for s in ORDER:
        d = per_seed[per_seed.strategy == s].pivot(index="length", columns="seed", values="score")
        m = d.mean(axis=1)
        half = stats.t.ppf(0.975, d.shape[1] - 1) * d.std(axis=1, ddof=1) / np.sqrt(d.shape[1])
        ax.plot(d.index, m, color=COLOR[s], lw=1.4, label=LABEL[s])
        ax.fill_between(d.index, (m - half).clip(lower=0), m + half, color=COLOR[s], alpha=0.10, lw=0)
    ax.set_xlabel("Pole length")
    ax.set_ylabel("Mean episode length (steps)")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "per_length")


def fig_phases(df):
    """Share of test episodes ending in each test phase, per strategy."""
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    shades = {1: "#d6d5d0", 2: "#8a8985", 3: "#3d3c3a"}
    names = {1: "ends in phase 1 (nominal)", 2: "ends in phase 2 (wind, force 75)",
             3: "reaches phase 3 (rising force)"}
    y = np.arange(len(ORDER))[::-1]
    for yi, s in zip(y, ORDER):
        frac = df[df.strategy == s].phase.value_counts(normalize=True)
        left = 0.0
        for ph in (1, 2, 3):
            w = frac.get(ph, 0.0)
            ax.barh(yi, w, left=left, color=shades[ph], height=0.62,
                    edgecolor="white", linewidth=1.0, label=names[ph] if yi == y[0] else None)
            if w > 0.07:
                ax.text(left + w / 2, yi, f"{100 * w:.0f}%", ha="center", va="center",
                        fontsize=6.5, color="white" if ph > 1 else "#0b0b0b")
            left += w
    ax.set_yticks(y, [SHORT[s] for s in ORDER])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100%"])
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    save(fig, "phases")


def fig_bars(df, res):
    """Assignment-style bar plot (avg +- std per length) for each approach,
    aggregated over all seeds and episodes; title = overall average."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
    for ax, s in zip(axes.flat, ORDER):
        d = df[df.strategy == s].groupby("length").score
        m, sd = d.mean(), d.std(ddof=0)
        x = np.arange(len(m))
        ax.bar(x, m.values, yerr=sd.values, color=COLOR[s], alpha=0.85, width=0.8,
               error_kw={"elinewidth": 0.5, "capsize": 1.5, "ecolor": "#52514e"})
        ax.set_xticks(x[::3], [f"{v:.2f}" for v in m.index[::3]], rotation=45)
        ax.set_title(f"{LABEL[s]} (10 seeds)\nAverage score over all pole lengths = {m.mean():.0f}")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Pole length")
        ax.grid(axis="y")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode length")
    fig.tight_layout()
    save(fig, "bar_plots_all")


def fig_training():
    """Training return (smoothed) and validation return over env steps."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9))
    grid = np.linspace(0, 200_000, 101)
    for s in ORDER:
        curves = []
        for path in glob.glob(os.path.join(RES, "logs", f"{s}_seed*.csv")):
            d = pd.read_csv(path)
            sm = d["return"].rolling(20, min_periods=1).mean()
            curves.append(np.interp(grid, d.step, sm))
        if curves:
            c = np.array(curves)
            axes[0].plot(grid, c.mean(0), color=COLOR[s], lw=1.2, label=LABEL[s])
            axes[0].fill_between(grid, c.min(0), c.max(0), color=COLOR[s], alpha=0.12, lw=0)
        vals = []
        for path in glob.glob(os.path.join(RES, "logs", f"{s}_seed*.json")):
            v = json.load(open(path))["validation"]
            vals.append([e["val_return"] for e in v])
        if vals:
            v = np.array(vals)
            steps = np.arange(1, v.shape[1] + 1) * 5_000
            axes[1].plot(steps, v.mean(0), color=COLOR[s], lw=1.2, marker="o", ms=2.5)
    axes[0].set_title("Training episode return (20-ep. moving avg., cap 1000)")
    axes[1].set_title("Validation return (nominal dynamics)")
    for ax in axes:
        ax.set_xlabel("Environment steps")
        ax.xaxis.set_major_formatter(KSTEPS)
        ax.grid(axis="y")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "training")


def fig_curriculum():
    """Sampling probability of each pole-length bin over training (S2)."""
    probs = []
    for path in glob.glob(os.path.join(RES, "logs", "adaptive_seed*.json")):
        v = json.load(open(path))["validation"]
        probs.append([e["sampler"]["probs"] for e in v])
    if not probs:
        return
    p = np.mean(np.array(probs), axis=0).T  # bins x checkpoints
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    im = ax.imshow(p, aspect="auto", origin="lower", cmap="Blues",
                   extent=[2_500, 202_500, 0.4, 1.8])
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Pole length bin")
    ax.xaxis.set_major_formatter(KSTEPS)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Sampling prob.", fontsize=7)
    save(fig, "curriculum")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    df = load_eval()
    models = per_model(df)
    models.to_csv(os.path.join(RES, "summary_per_model.csv"), index=False)
    res = statistics(models, df)
    write_stats(res)
    fig_per_length(df)
    fig_phases(df)
    fig_bars(df, res)
    fig_training()
    fig_curriculum()
    print("figures written to", FIG)
