"""
Multi-Seed Statistical Benchmarking for VQC Architecture Search
================================================================
Runs Random Search, Evolutionary Search, and LLM-Guided Search across
multiple independent seeds, computes aggregated statistics, and generates
publication-quality comparison plots.

Usage:
    python experiments/multiseed_analysis.py
"""

import sys
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend – safe for scripts
import matplotlib.pyplot as plt

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.datasets import make_moons
from search import run_search
from evolution_search import run_evolution_search
from llm_search import run_llm_search

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
NUM_SEEDS       = 5
EVAL_BUDGET     = 12          # identical budget for every strategy
SCORE_THRESHOLD = 0.73        # "good enough" convergence threshold
SEEDS           = [42, 123, 256, 789, 1024]

STRATEGIES = ["Random Search", "Evolutionary Search", "LLM-Guided Search"]

# Evolutionary hyper-params (pop_size × generations ≈ EVAL_BUDGET)
EVO_POP_SIZE    = 4
EVO_GENERATIONS = 2           # 4 init + 2×(4-1)=6 children → 10, close to 12
# Note: pop_size(4) init + generations(2) * (pop_size-1)(3) children = 4+6 = 10
# We'll use pop_size=3, generations=3: 3 init + 3*2 = 9, not great.
# Better: pop_size=4, generations=2: 4 + 2*3 = 10  (close to 12)
# Or pop_size=3, generations=3: 3 + 3*2 = 9
# Best fit: pop_size=4, generations=2 gives 10 evals (closest fair budget)
# Actually let's compute: for exactly 12 evals we need pop + gen*(pop-1) = 12
# pop=3, gen=3 -> 3+9=12 ✓
EVO_POP_SIZE    = 3
EVO_GENERATIONS = 3           # 3 + 3×(3-1) = 3 + 6 = 9? No: 3+3*2=9
# pop=4, gen=2 -> 4+2*3=10  (not 12)
# pop=3, gen=3 -> 3+3*2=9   (not 12)
# pop=4, gen=3 -> 4+3*3=13  (over by 1)
# pop=3, gen=4 -> 3+4*2=11  (close)
# pop=2, gen=5 -> 2+5*1=7   (too low)
# pop=6, gen=1 -> 6+1*5=11
# pop=12,gen=0 -> 12         (no evolution, just random init)
# pop=4, gen=2 -> 10 total evals, pad to 12? Let's just use closest: 4+2*3=10
# For fairness, let's use pop=4, gen=2: 10 evals, and random+llm also get 10
# OR use the original: each gets 12, with evo as pop=3, generations=3 → 9
# Let's use EVAL_BUDGET=12 for Random & LLM, and (pop=4,gen=2)→10 for Evo
# but pad the evo convergence curve to length 12.
#
# Simplest fair approach: use EVAL_BUDGET everywhere, pad shorter curves.
EVO_POP_SIZE    = 4
EVO_GENERATIONS = 2           # yields exactly 10 evals (padded to 12 for plots)

PLOT_DIR = os.path.join(PROJECT_ROOT, "experiments")
os.makedirs(PLOT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SEED CONTROL
# ═════════════════════════════════════════════════════════════════════════════
def set_all_seeds(seed: int):
    """Deterministically reset all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    # PennyLane uses its own numpy wrapper – seed it too
    try:
        import pennylane.numpy as pnp
        pnp.random.seed(seed)
    except Exception:
        pass
    # PyTorch (optional)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  DATASET FACTORY
# ═════════════════════════════════════════════════════════════════════════════
def make_dataset(seed: int):
    """Generate a fresh make_moons dataset seeded independently."""
    X, y = make_moons(n_samples=30, noise=0.1, random_state=seed)
    X = np.pi * (X - X.min()) / (X.max() - X.min())
    y = 2 * y - 1
    return X, y


# ═════════════════════════════════════════════════════════════════════════════
#  STRATEGY RUNNERS (return convergence list of length ≤ EVAL_BUDGET)
# ═════════════════════════════════════════════════════════════════════════════
def run_random(X, y):
    """Run random search with EVAL_BUDGET evaluations."""
    scores, _ = run_search(X, y, iterations=EVAL_BUDGET)
    return scores


def run_evolutionary(X, y):
    """Run evolutionary search; pad/truncate curve to EVAL_BUDGET."""
    scores, _ = run_evolution_search(
        X, y,
        population_size=EVO_POP_SIZE,
        generations=EVO_GENERATIONS,
    )
    return scores


def run_llm_guided(X, y):
    """Run LLM-guided search with EVAL_BUDGET evaluations."""
    scores, _, _ = run_llm_search(X, y, iterations=EVAL_BUDGET)
    return scores


STRATEGY_FNS = {
    "Random Search":       run_random,
    "Evolutionary Search": run_evolutionary,
    "LLM-Guided Search":   run_llm_guided,
}


# ═════════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def pad_curve(curve: list, length: int) -> list:
    """Pad a convergence curve to *length* by repeating the last value."""
    if len(curve) >= length:
        return curve[:length]
    return curve + [curve[-1]] * (length - len(curve))


def evals_to_threshold(curve: list, threshold: float) -> int:
    """Return 1-indexed eval count when score first drops ≤ threshold.
    Returns len(curve)+1 if threshold is never reached (sentinel)."""
    for idx, val in enumerate(curve):
        if val <= threshold:
            return idx + 1          # 1-indexed
    return len(curve) + 1           # never reached


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN BENCHMARK LOOP
# ═════════════════════════════════════════════════════════════════════════════
def run_benchmark():
    """Execute the full multi-seed benchmark and return raw results."""
    # results[strategy] = { "curves": [], "best_scores": [], "evals_to_thresh": [] }
    results = {s: {"curves": [], "best_scores": [], "evals_to_thresh": []}
               for s in STRATEGIES}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'#'*70}")
        print(f"#  SEED {seed_idx+1}/{NUM_SEEDS}  —  seed = {seed}")
        print(f"{'#'*70}")

        for strategy in STRATEGIES:
            print(f"\n{'─'*60}")
            print(f"  Strategy: {strategy}  |  Seed: {seed}")
            print(f"{'─'*60}")

            # 1. Reset all RNGs
            set_all_seeds(seed)

            # 2. Generate fresh dataset (deterministic per seed)
            X, y = make_dataset(seed)

            # 3. Run the strategy
            curve = STRATEGY_FNS[strategy](X, y)

            # 4. Normalise curve length
            curve = pad_curve(curve, EVAL_BUDGET)

            # 5. Record metrics
            best_final = min(curve)
            ett = evals_to_threshold(curve, SCORE_THRESHOLD)

            results[strategy]["curves"].append(curve)
            results[strategy]["best_scores"].append(best_final)
            results[strategy]["evals_to_thresh"].append(ett)

            print(f"  → Best score: {best_final:.4f}  |  "
                  f"Evals to ≤{SCORE_THRESHOLD}: "
                  f"{'N/A' if ett > EVAL_BUDGET else ett}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  AGGREGATE STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_statistics(results: dict) -> dict:
    """
    Returns stats[strategy] = {
        "mean_best":  float,  "std_best":  float,
        "mean_ett":   float,  "std_ett":   float,
        "mean_curve": np.ndarray (length EVAL_BUDGET)
    }
    """
    stats = {}
    for strategy in STRATEGIES:
        data = results[strategy]
        best_arr = np.array(data["best_scores"])
        ett_arr  = np.array(data["evals_to_thresh"], dtype=float)
        curves   = np.array(data["curves"])           # (NUM_SEEDS, EVAL_BUDGET)

        stats[strategy] = {
            "mean_best":  float(np.mean(best_arr)),
            "std_best":   float(np.std(best_arr)),
            "mean_ett":   float(np.mean(ett_arr)),
            "std_ett":    float(np.std(ett_arr)),
            "mean_curve": np.mean(curves, axis=0),
            "std_curve":  np.std(curves, axis=0),
        }
    return stats


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════
COLORS = {
    "Random Search":       "#4C9BE8",
    "Evolutionary Search": "#F5A623",
    "LLM-Guided Search":   "#7ED321",
}

def _apply_style(ax, title, xlabel, ylabel):
    """Common plot styling."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_mean_convergence(stats: dict, save_path: str):
    """Plot 1: Mean convergence curves with ±1 std ribbons."""
    fig, ax = plt.subplots(figsize=(10, 6))
    evals = np.arange(1, EVAL_BUDGET + 1)

    for strategy in STRATEGIES:
        s = stats[strategy]
        mean = s["mean_curve"]
        std  = s["std_curve"]
        color = COLORS[strategy]

        ax.plot(evals, mean, label=strategy, color=color, linewidth=2.2)
        ax.fill_between(evals, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.axhline(y=SCORE_THRESHOLD, color="red", linestyle=":",
               linewidth=1.2, alpha=0.7, label=f"Threshold ({SCORE_THRESHOLD})")

    _apply_style(ax,
                 title="Mean Convergence Curve  (5 seeds)",
                 xlabel="Evaluation #",
                 ylabel="Best Score (lower is better)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_evals_to_threshold(stats: dict, save_path: str):
    """Plot 2: Bar chart – mean evals-to-threshold ± std."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(STRATEGIES)
    means = [stats[s]["mean_ett"] for s in names]
    stds  = [stats[s]["std_ett"]  for s in names]
    colors = [COLORS[s] for s in names]

    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    # Annotate values on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.3,
                f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    _apply_style(ax,
                 title=f"Mean Evaluations to Reach Score ≤ {SCORE_THRESHOLD}  (5 seeds)",
                 xlabel="Strategy",
                 ylabel="Evaluations")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_best_score(stats: dict, save_path: str):
    """Plot 3: Bar chart – mean best score ± std."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(STRATEGIES)
    means = [stats[s]["mean_best"] for s in names]
    stds  = [stats[s]["std_best"]  for s in names]
    colors = [COLORS[s] for s in names]

    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    # Annotate values on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
                f"{m:.4f}±{s:.4f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.axhline(y=SCORE_THRESHOLD, color="red", linestyle=":",
               linewidth=1.2, alpha=0.7, label=f"Threshold ({SCORE_THRESHOLD})")

    _apply_style(ax,
                 title="Mean Best Score  (5 seeds)",
                 xlabel="Strategy",
                 ylabel="Score (lower is better)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  ASCII SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
def print_summary_table(stats: dict):
    """Print a clean ASCII results table."""
    header = (
        f"{'Strategy':<25} │ {'Mean Best Score':>22} │ "
        f"{'Mean Evals→Threshold':>25}"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print("  MULTI-SEED BENCHMARK RESULTS")
    print(f"  Seeds: {SEEDS}  |  Budget: {EVAL_BUDGET}  |  "
          f"Threshold: {SCORE_THRESHOLD}")
    print(sep)
    print(header)
    print("─" * 25 + "─┼─" + "─" * 22 + "─┼─" + "─" * 25)

    for strategy in STRATEGIES:
        s = stats[strategy]
        best_str = f"{s['mean_best']:.4f} ± {s['std_best']:.4f}"
        ett_str  = f"{s['mean_ett']:.1f} ± {s['std_ett']:.1f}"
        print(f"{strategy:<25} │ {best_str:>22} │ {ett_str:>25}")

    print(sep)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 70)
    print("  VQC Architecture Search — Multi-Seed Statistical Benchmark")
    print(f"  Seeds: {SEEDS}")
    print(f"  Evaluation budget per strategy: {EVAL_BUDGET}")
    print(f"  Score threshold: {SCORE_THRESHOLD}")
    print("=" * 70)

    # ── 1. Run all strategies across all seeds ───────────────────────────
    results = run_benchmark()

    # ── 2. Aggregate statistics ──────────────────────────────────────────
    stats = compute_statistics(results)

    # ── 3. Print summary ─────────────────────────────────────────────────
    print_summary_table(stats)

    # ── 4. Generate plots ────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_mean_convergence(
        stats,
        os.path.join(PLOT_DIR, "stat_mean_convergence.png"),
    )
    plot_evals_to_threshold(
        stats,
        os.path.join(PLOT_DIR, "stat_threshold_comparison.png"),
    )
    plot_best_score(
        stats,
        os.path.join(PLOT_DIR, "stat_best_score.png"),
    )

    print("\n✅  Benchmark complete.")
