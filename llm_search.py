from circuit_builder import build_circuit
from trainer import train_architecture
from evaluator import evaluate_structure, compute_score
from llm_agent import llm_generate_architecture


def run_llm_search(X, y, iterations=6):
    """
    Feedback-driven LLM agent search loop.

    Each iteration:
      1. Agent receives full history of past proposals + scores
      2. Agent reasons about what to propose next (conditioning on feedback)
      3. Circuit is built, trained, evaluated
      4. Result appended to history for next iteration

    Returns:
        best_scores:  convergence curve — best score after each evaluation.
        best:         dict with keys "architecture", "loss", "score".
        history:      full record of all proposals and outcomes.
    """

    history = []
    best_scores = []
    best_so_far = float("inf")

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
        score = float(compute_score(loss, depth, cnot))

        print(f"Loss:  {float(loss):.4f}")
        print(f"Depth: {depth}  |  CNOT: {cnot}  |  Score: {score:.4f}")

        # ── Track convergence ─────────────────────────────────────────────
        if score < best_so_far:
            best_so_far = score
        best_scores.append(best_so_far)

        # ── Append to history (feedback into next agent call) ────────────
        history.append({
            "architecture": arch,
            "loss": float(loss),
            "score": score
        })

    best = min(history, key=lambda x: x["score"])

    print(f"\n{'='*50}")
    print(f"  Agent search complete. Best score: {best['score']:.4f}")
    print(f"{'='*50}")

    return best_scores, best, history
