import numpy as np
from sklearn.datasets import make_moons

from llm_search import run_llm_search

# ── Dataset ──────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=30, noise=0.1)

# Scale features to [0, π]
X = np.pi * (X - X.min()) / (X.max() - X.min())

# Convert labels to {-1, +1}
y = 2 * y - 1

# ── LLM-Guided Search ────────────────────────────────────────────────────────
best, history = run_llm_search(X, y, iterations=6)

print("\n===== BEST (LLM-GUIDED) =====")
print(f"  Architecture : {best['architecture']}")
print(f"  Loss         : {best['loss']:.4f}")
print(f"  Score        : {best['score']:.4f}")

print("\n===== FULL AGENT HISTORY =====")
for i, entry in enumerate(history):
    print(f"  [{i+1}] score={entry['score']:.4f}  loss={entry['loss']:.4f}  "
          f"layers={entry['architecture']['n_layers']}  "
          f"entanglement={entry['architecture']['entanglement']}")