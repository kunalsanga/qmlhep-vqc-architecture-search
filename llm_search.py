import json
import os
from circuit_builder import build_circuit
from trainer import train_architecture
from evaluator import evaluate_structure, compute_score
from llm_agent import llm_generate_architecture, propose_architecture_with_trace, rule_based_generate_architecture


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
        arch, prompt, raw_response = propose_architecture_with_trace(history)
        print("Proposed Architecture:", arch)

        # ── Build, train, evaluate ───────────────────────────────────────
        circuit = build_circuit(arch)
        loss = train_architecture(circuit, arch, X, y)
        depth, total_gates, cnot = evaluate_structure(arch)
        score = float(compute_score(loss, depth, cnot))

        print(f"Loss:  {float(loss):.4f}")
        print(f"Depth: {depth}  |  CNOT: {cnot}  |  Score: {score:.4f}")

        # ── Track Trace ───────────────────────────────────────────────────
        trace_entry = {
            "iteration": i + 1,
            "prompt": prompt,
            "raw_output": raw_response,
            "parsed_architecture": arch,
            "final_score": score
        }
        trace_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "llm_reasoning_trace.json")
        
        trace_data = []
        if os.path.exists(trace_file):
            try:
                with open(trace_file, "r") as f:
                    trace_data = json.load(f)
            except Exception:
                pass
        trace_data.append(trace_entry)
        with open(trace_file, "w") as f:
            json.dump(trace_data, f, indent=2)

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

def run_rule_based_search(X, y, iterations=6):
    """
    Rule-based (Heuristic) LLM agent search loop.
    """

    history = []
    best_scores = []
    best_so_far = float("inf")

    for i in range(iterations):

        print(f"\n{'='*50}")
        print(f"  Rule-Based Iteration {i+1} / {iterations}")
        print(f"{'='*50}")

        arch = rule_based_generate_architecture(history)
        print("Proposed Architecture:", arch)

        circuit = build_circuit(arch)
        loss = train_architecture(circuit, arch, X, y)
        depth, total_gates, cnot = evaluate_structure(arch)
        score = float(compute_score(loss, depth, cnot))

        print(f"Loss:  {float(loss):.4f}")
        print(f"Depth: {depth}  |  CNOT: {cnot}  |  Score: {score:.4f}")

        if score < best_so_far:
            best_so_far = score
        best_scores.append(best_so_far)

        history.append({
            "architecture": arch,
            "loss": float(loss),
            "score": score
        })

    best = min(history, key=lambda x: x["score"])

    return best_scores, best, history
