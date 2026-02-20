import numpy as np
from sklearn.datasets import make_moons

from evolution_search import run_evolution_search
from plots import plot_results

# Dataset
X, y = make_moons(n_samples=30, noise=0.1)

# Scale inputs to [-pi, pi]
X = np.pi * (X - X.min()) / (X.max() - X.min())

# Convert labels to -1 and +1
y = 2 * y - 1

# Run evolutionary search
best = run_evolution_search(
    X,
    y,
    population_size=4,   # small population
    generations=4        # small generations
)

print("\n===== FINAL BEST (EVOLUTION) =====")
print(best)