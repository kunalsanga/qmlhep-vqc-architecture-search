import sys
import os
import numpy as np
import random

# Ensure project modules can be found
sys.path.append(os.path.abspath('..'))

try:
    from sklearn.datasets import make_moons
except ImportError:
    print("sklearn not found. Please install scikit-learn.")
    sys.exit(1)

from llm_search import run_llm_search, run_rule_based_search

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def evaluations_to_threshold(scores, threshold=0.73):
    for count, score in enumerate(scores, 1):
        if score <= threshold:
            return count
    return len(scores)

def run_multiple_seeds(strategy_fn, strategy_name, n_runs=2, iterations=15, threshold=0.73):
    print(f"\n--- Running {strategy_name} across {n_runs} seeds ---")
    eval_counts = []
    
    for seed in range(n_runs):
        set_seed(seed)
        X, y = make_moons(n_samples=20, noise=0.1, random_state=seed)
        y = np.where(y == 0, -1, 1)
        
        print(f"[{strategy_name}] Seed {seed+1}/{n_runs}...")
        best_scores, _, _ = strategy_fn(X, y, iterations=iterations)
        
        evals = evaluations_to_threshold(best_scores, threshold)
        eval_counts.append(evals)
        
    return eval_counts

N_RUNS = 2
MAX_EVALS = 10
THRESHOLD = 0.73

print("Starting evaluation comparison...")
rule_evals = run_multiple_seeds(run_rule_based_search, "Rule-Based Agent", n_runs=N_RUNS, iterations=MAX_EVALS, threshold=THRESHOLD)
llm_evals = run_multiple_seeds(run_llm_search, "Real LLM Agent", n_runs=N_RUNS, iterations=MAX_EVALS, threshold=THRESHOLD)

mean_rule = np.mean(rule_evals)
mean_llm = np.mean(llm_evals)

print("\n" + "="*50)
print(f"Mean Evals (Rule-Based Agent): {mean_rule:.2f}")
print(f"Mean Evals (Real LLM Agent):   {mean_llm:.2f}")
if mean_rule > mean_llm:
    improvement = ((mean_rule - mean_llm) / mean_rule) * 100
    print(f"Real LLM-guided search achieved {improvement:.1f}% improvement.")
elif mean_rule < mean_llm:
    decline = ((mean_llm - mean_rule) / mean_rule) * 100
    print(f"Real LLM-guided search was {decline:.1f}% slower to converge.")
else:
    print("Real LLM-guided search showed identical performance.")
print("="*50)
