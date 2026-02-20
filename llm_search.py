from circuit_builder import build_circuit
from trainer import train_architecture
from evaluator import evaluate_structure, compute_score
from llm_agent import llm_generate_architecture


def run_llm_search(X, y, iterations=5):
    """
    Feedback-driven LLM agent search loop.

    Each iteration:
      1. Agent receives full history of past proposals + scores
      2. Agent reasons about what to propose next
      3. Circuit is built, trained, evaluated
      4. Result is appended to history for next iteration

    This is a closed-loop, feedback-conditioned architecture search.
    The agent improves its proposals based on observed outcomes.

    Args:
        X: input features, shape (n_samples, n_features)
        y: labels in {-1, +1}, shape (n_samples,)
        iterations: number of agent proposal cycles

    Returns:
        best: dict with keys "architecture", "loss", "score"
    """

    history = []

    for i in range(iterations):

        print(f"\n{'='*50}")
        print(f"  LLM Iteration {i+1} / {iterations}")
        print(f"{'='*50}")

        # ── Agent proposes architecture based on history ─────────────────
        arch = llm_generate_architecture(history)
        print("Proposed Architecture:", arch)

        # ── Build, train, evaluate ───────────────────────────────────────
        circuit = build_circuit(arch)
        loss = train_architecture(circuit, arch, X, y)
        depth, total_gates, cnot = evaluate_structure(arch)
        score = compute_score(loss, depth, cnot)

        print(f"Loss:  {float(loss):.4f}")
        print(f"Depth: {depth}  |  CNOT: {cnot}  |  Score: {float(score):.4f}")

        # ── Append result to history (feedback) ──────────────────────────
        history.append({
            "architecture": arch,
            "loss": float(loss),
            "score": float(score)
        })

    # ── Select best across all iterations ───────────────────────────────
    best = min(history, key=lambda x: x["score"])

    print(f"\n{'='*50}")
    print(f"  Agent search complete. Best score: {best['score']:.4f}")
    print(f"{'='*50}")

    return best, history
