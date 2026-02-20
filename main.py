import numpy as np
from sklearn.datasets import make_moons

from search import run_search
from evolution_search import run_evolution_search
from llm_search import run_llm_search
from comparison_plot import plot_comparison

# ── Dataset ───────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=30, noise=0.1)
X = np.pi * (X - X.min()) / (X.max() - X.min())
y = 2 * y - 1

# ── Run all three search strategies ──────────────────────────────────────────
print("\n" + "="*60)
print("  STAGE 1: Random Search")
print("="*60)
random_scores, _ = run_search(X, y, iterations=8)

print("\n" + "="*60)
print("  STAGE 2: Evolutionary Search")
print("="*60)
evolution_scores, _ = run_evolution_search(X, y, population_size=4, generations=4)

print("\n" + "="*60)
print("  STAGE 3: LLM-Guided Search")
print("="*60)
llm_scores, best, history = run_llm_search(X, y, iterations=6)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)
print(f"  Random Search     →  best score: {min(random_scores):.4f}  "
      f"({len(random_scores)} evals)")
print(f"  Evolutionary      →  best score: {min(evolution_scores):.4f}  "
      f"({len(evolution_scores)} evals)")
print(f"  LLM-Guided        →  best score: {min(llm_scores):.4f}  "
      f"({len(llm_scores)} evals)")

print(f"\n  LLM best architecture: {best['architecture']}")

# ── Comparison Plot ───────────────────────────────────────────────────────────
plot_comparison(random_scores, evolution_scores, llm_scores)