import csv
from architecture import generate_architecture
from circuit_builder import build_circuit
from trainer import train_architecture
from evaluator import evaluate_structure, compute_score


def run_search(X, y, iterations=10):
    """
    Random search — evaluates N independently sampled architectures.

    Returns:
        best_scores: list of best score seen so far at each evaluation step.
                     Represents the convergence curve for plotting.
        results:     full list of result dicts (for CSV export).
    """

    best_scores = []
    best_so_far = float("inf")
    results = []

    for i in range(iterations):

        print(f"\n===== Iteration {i+1} =====")

        arch = generate_architecture()
        print("Architecture:", arch)

        circuit = build_circuit(arch)
        loss = train_architecture(circuit, arch, X, y)

        depth, total_gates, cnot = evaluate_structure(arch)
        score = float(compute_score(loss, depth, cnot))

        print(f"Loss: {float(loss):.4f}  |  Depth: {depth}  |  "
              f"CNOT: {cnot}  |  Score: {score:.4f}")

        results.append({
            "architecture": arch,
            "loss": float(loss),
            "depth": depth,
            "cnot": cnot,
            "score": score
        })

        # Track best so far (convergence curve)
        if score < best_so_far:
            best_so_far = score
        best_scores.append(best_so_far)

    # Save full results to CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["loss", "depth", "cnot", "score", "architecture"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    return best_scores, results