import numpy as np
from sklearn.datasets import make_moons
from search import run_search

# Dataset
X, y = make_moons(n_samples=30, noise=0.1)

X = np.pi * (X - X.min()) / (X.max() - X.min())
y = 2*y - 1

best, results = run_search(X, y, iterations=8)

print("\n\n===== BEST ARCHITECTURE FOUND =====")
print(best)