import csv
from architecture import generate_architecture
from circuit_builder import build_circuit
from trainer import train_architecture
from evaluator import evaluate_structure, compute_score


def run_search(X, y, iterations=10):

    results = []

    for i in range(iterations):

        print(f"\n===== Iteration {i+1} =====")

        arch = generate_architecture()
        print("Architecture:", arch)

        circuit = build_circuit(arch)

        loss = train_architecture(circuit, arch, X, y)

        depth, total_gates, cnot = evaluate_structure(arch)
        score = compute_score(loss, depth, cnot)

        print("Loss:", float(loss))
        print("Depth:", depth)
        print("CNOT:", cnot)
        print("Score:", float(score))

        results.append({
            "architecture": arch,
            "loss": float(loss),
            "depth": depth,
            "cnot": cnot,
            "score": float(score)
        })

    # Select best architecture
    best = min(results, key=lambda x: x["score"])

        # Save results to CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["loss", "depth", "cnot", "score", "architecture"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    return best, results