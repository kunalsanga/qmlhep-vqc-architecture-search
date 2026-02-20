from architecture import generate_architecture
from circuit_builder import build_circuit
from trainer import train_architecture
from evaluator import evaluate_structure, compute_score
from evolution import mutate_architecture


def run_evolution_search(X, y, population_size=4, generations=4):
    """
    Elitist (1+λ) evolutionary search.

    Each generation: selects best from population, mutates it to fill
    the next generation. Tracks one score per evaluation for a fair
    comparison curve against random and LLM-guided search.

    Returns:
        best_scores: convergence curve — best score after each circuit eval.
        final_best:  dict with keys "architecture", "loss", "score".
    """

    best_scores = []
    best_so_far = float("inf")

    # ── Initial random population ────────────────────────────────────────
    population = []

    print("\n[Evolution] Initialising population...")
    for _ in range(population_size):
        arch = generate_architecture()
        circuit = build_circuit(arch)
        loss = train_architecture(circuit, arch, X, y)

        depth, total_gates, cnot = evaluate_structure(arch)
        score = float(compute_score(loss, depth, cnot))

        population.append({
            "architecture": arch,
            "score": score,
            "loss": float(loss)
        })

        if score < best_so_far:
            best_so_far = score
        best_scores.append(best_so_far)

    # ── Evolution loop ───────────────────────────────────────────────────
    for gen in range(generations):

        print(f"\n=== Generation {gen+1} ===")

        # Select best (elitism)
        best = min(population, key=lambda x: x["score"])
        print(f"Best score: {best['score']:.4f}")

        # Mutate best to create new candidates
        new_population = [best]

        for _ in range(population_size - 1):
            mutated_arch = mutate_architecture(best["architecture"])
            circuit = build_circuit(mutated_arch)
            loss = train_architecture(circuit, mutated_arch, X, y)

            depth, total_gates, cnot = evaluate_structure(mutated_arch)
            score = float(compute_score(loss, depth, cnot))

            new_population.append({
                "architecture": mutated_arch,
                "score": score,
                "loss": float(loss)
            })

            # Track convergence across ALL evaluations (including children)
            if score < best_so_far:
                best_so_far = score
            best_scores.append(best_so_far)

        population = new_population

    final_best = min(population, key=lambda x: x["score"])

    return best_scores, final_best