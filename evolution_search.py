from architecture import generate_architecture
from circuit_builder import build_circuit
from trainer import train_architecture
from evaluator import evaluate_structure, compute_score
from evolution import mutate_architecture


def run_evolution_search(X, y, population_size=5, generations=5):

    # Initial random population
    population = []

    for _ in range(population_size):
        arch = generate_architecture()
        circuit = build_circuit(arch)
        loss = train_architecture(circuit, arch, X, y)

        depth, total_gates, cnot = evaluate_structure(arch)
        score = compute_score(loss, depth, cnot)

        population.append({
            "architecture": arch,
            "score": float(score),
            "loss": float(loss)
        })

    # Evolution loop
    for gen in range(generations):

        print(f"\n=== Generation {gen+1} ===")

        # Select best
        best = min(population, key=lambda x: x["score"])
        print("Best score:", best["score"])

        # Mutate best to create new candidates
        new_population = [best]

        for _ in range(population_size - 1):
            mutated_arch = mutate_architecture(best["architecture"])
            circuit = build_circuit(mutated_arch)
            loss = train_architecture(circuit, mutated_arch, X, y)

            depth, total_gates, cnot = evaluate_structure(mutated_arch)
            score = compute_score(loss, depth, cnot)

            new_population.append({
                "architecture": mutated_arch,
                "score": float(score),
                "loss": float(loss)
            })

        population = new_population

    final_best = min(population, key=lambda x: x["score"])

    return final_best