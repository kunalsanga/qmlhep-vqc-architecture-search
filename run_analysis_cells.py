import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Append parent directory to sys.path so project modules can be imported
sys.path.append(os.path.abspath('..'))

from search import run_search
from evolution_search import run_evolution_search
from llm_search import run_llm_search, run_rule_based_search
from config import MAX_LAYERS, ALLOWED_ROTATIONS, ENTANGLEMENT_PATTERNS

# Ensure reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

def evaluations_to_threshold(scores, threshold=0.73):
    """
    Returns the number of evaluations required to reach score <= threshold.
    If threshold is not reached, returns the total number of evaluations.
    """
    for count, score in enumerate(scores, 1):
        if score <= threshold:
            return count
    return len(scores)

def run_multiple_seeds(strategy_fn, strategy_name, n_runs=5, iterations=20, threshold=0.73):
    """
    Runs a search strategy across different seeds,
    generating a new dataset for each seed.
    """
    print(f"\n--- Running {strategy_name} across {n_runs} seeds ---")
    eval_counts = []
    all_curves = []
    
    for seed in range(n_runs):
        set_seed(seed)
        
        # Generate dataset
        X, y = make_moons(n_samples=20, noise=0.1, random_state=seed)
        y = np.where(y == 0, -1, 1)  # Scale for PauliZ expectation [-1, 1]
        
        print(f"[{strategy_name}] Seed {seed+1}/{n_runs}...")
        
        if strategy_name == "Random":
            best_scores, _ = strategy_fn(X, y, iterations=iterations)
        elif strategy_name == "Evolutionary":
            # Evolutionary evaluates population_size per generation
            # initial: 4, per gen: 3 (generations=6) => total 22 evaluations.
            # cap best_scores at iterations (20) elements
            best_scores, _ = strategy_fn(X, y, population_size=4, generations=6)
            best_scores = best_scores[:iterations]
        elif strategy_name == "Rule-Based LLM":
            best_scores, _, _ = strategy_fn(X, y, iterations=iterations)
        elif strategy_name == "LLM":
            best_scores, _, _ = strategy_fn(X, y, iterations=iterations)
            
        evals = evaluations_to_threshold(best_scores, threshold)
        eval_counts.append(evals)
        
        # Normalize curve length to exact number of iterations
        curve = best_scores[:iterations]
        if len(curve) < iterations:
            curve.extend([curve[-1]] * (iterations - len(curve)))
            
        all_curves.append(curve)
        
    return eval_counts, np.array(all_curves)

# Parameters
N_RUNS = 5
MAX_EVALS = 20
THRESHOLD = 0.73

results = {}

# Run Strategies
rand_evals, rand_curves = run_multiple_seeds(run_search, "Random", n_runs=N_RUNS, iterations=MAX_EVALS, threshold=THRESHOLD)
results["Random"] = {"evals": rand_evals, "curves": rand_curves}

evo_evals, evo_curves = run_multiple_seeds(run_evolution_search, "Evolutionary", n_runs=N_RUNS, iterations=MAX_EVALS, threshold=THRESHOLD)
results["Evolutionary"] = {"evals": evo_evals, "curves": evo_curves}

rb_evals, rb_curves = run_multiple_seeds(run_rule_based_search, "Rule-Based LLM", n_runs=N_RUNS, iterations=MAX_EVALS, threshold=THRESHOLD)
results["Rule-Based LLM"] = {"evals": rb_evals, "curves": rb_curves}

llm_evals, llm_curves = run_multiple_seeds(run_llm_search, "LLM", n_runs=N_RUNS, iterations=MAX_EVALS, threshold=THRESHOLD)
results["LLM"] = {"evals": llm_evals, "curves": llm_curves}

# Print Summary Table
print("\n" + "="*50)
print(f"{'Strategy':<15} | {'Mean Evals':<12} | {'Std Dev':<10} | {'Variance':<10}")
print("-" * 50)

for strategy, data in results.items():
    evals = data["evals"]
    mean_evals = np.mean(evals)
    std_dev = np.std(evals)
    variance = np.var(evals)
    
    print(f"{strategy:<15} | {mean_evals:<12.2f} | {std_dev:<10.2f} | {variance:<10.2f}")

print("="*50)

strategies = list(results.keys())
mean_evals = [np.mean(results[s]["evals"]) for s in strategies]
std_evals = [np.std(results[s]["evals"]) for s in strategies]

# 1. Bar Chart
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
plt.bar(strategies, mean_evals, yerr=std_evals, capsize=10, color=colors, alpha=0.8, edgecolor='black')
plt.xlabel('Strategy', fontsize=12)
plt.ylabel(f'Mean Evaluations to Threshold ({THRESHOLD})', fontsize=12)
plt.title('Sample Efficiency Comparison', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('mean_evaluations.png', dpi=300)
plt.show()

# 2. Convergence Plot
plt.figure(figsize=(10, 6))

for strategy, color in zip(strategies, colors):
    curves = results[strategy]["curves"]
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    
    x = np.arange(1, MAX_EVALS + 1)
    plt.plot(x, mean_curve, label=strategy, color=color, linewidth=2)
    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)

plt.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD})')

plt.xlabel('Circuit Evaluations', fontsize=12)
plt.ylabel('Best Score So Far', fontsize=12)
plt.title('Average Convergence Curve Across Seeds', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('convergence_comparison.png', dpi=300)
plt.show()

# Given the configuration:
# MAX_LAYERS: max number of repeating layers
# ALLOWED_ROTATIONS: gate types per block
# ENTANGLEMENT_PATTERNS: available entanglement topologies

search_space_size = MAX_LAYERS * (len(ALLOWED_ROTATIONS) ** 2) * len(ENTANGLEMENT_PATTERNS)

print("====================================================")
print(f"Approximate Architecture Search Space Size (3 qubits): {search_space_size}")
print("====================================================\n")

print("Explanation of Exponential Scaling:")
print("The search space for variational quantum circuits scales exponentially with the number of qubits.")
print("Let Q = Number of Qubits")
print("    L = Maximum Layers")
print("    R = Allowed Rotation Gates")
print("    E = Entanglement Topologies")
print("\nIf we choose distinct rotation gates for qubits, the space roughly becomes:")
print("Space = L * ( |R|^Q ) * |E|")
print("\nWith Q=3, |R|=3, |E|=3, L=4: Space = 4 * (3^3) * 3 = 324 modular architectures.")
print("However, as Q increases to 10 (NISQ regimes), the rotation choices alone become 3^10 = 59,049 per block.")
print("Combined with varied layer structures, continuous parameter constraints, and depth penalties,\nnaive random or unguided evolutionary search quickly becomes physically intractable on near-term hardware.")

