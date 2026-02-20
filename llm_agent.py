import random
from config import ALLOWED_ROTATIONS, ENTANGLEMENT_PATTERNS, MAX_LAYERS


def llm_generate_architecture(history):
    """
    Simulates an LLM agent proposing a VQC architecture.

    In a real system, `history` would be serialized into a prompt and
    sent to an LLM (e.g. GPT-4, Claude). The LLM response would then
    be parsed back into an architecture dict.

    Here we simulate that reasoning loop with hand-coded heuristics
    that mirror the kind of logic a well-prompted LLM would apply:

      - No history → reasonable default (mid-depth, linear entanglement)
      - High loss   → increase layers (more expressivity needed)
      - Low loss    → keep structure, try varying entanglement
      - Very low score already → preserve architecture exactly

    Args:
        history: list of dicts, each with keys:
                 "architecture" (dict), "loss" (float), "score" (float)

    Returns:
        architecture dict with keys:
        n_qubits, n_layers, rotation_gates, entanglement
    """

    # ── Cold start: no history yet ──────────────────────────────────────
    if not history:
        print("[LLM-Agent] No history. Proposing default starting architecture.")
        return {
            "n_qubits": 3,
            "n_layers": 2,
            "rotation_gates": ["RY", "RZ"],
            "entanglement": "linear"
        }

    # ── Retrieve best seen so far ────────────────────────────────────────
    best = min(history, key=lambda x: x["score"])
    best_arch = best["architecture"]
    best_loss = best["loss"]
    best_score = best["score"]

    new_layers = best_arch["n_layers"]
    new_entanglement = best_arch["entanglement"]
    new_rotations = best_arch["rotation_gates"]

    # ── Reasoning rule 1: loss too high → need more expressivity ────────
    if best_loss > 0.65:
        new_layers = min(new_layers + 1, MAX_LAYERS)
        print(f"[LLM-Agent] Loss high ({best_loss:.3f}). Increasing layers → {new_layers}.")

    # ── Reasoning rule 2: loss ok but score high → too many CNOTs ───────
    elif best_loss <= 0.65 and best_score > 0.75:
        # Try reducing entanglement complexity to cut CNOT penalty
        if new_entanglement == "full":
            new_entanglement = "linear"
            print(f"[LLM-Agent] Score still high. Downgrading entanglement: full → linear.")
        elif new_entanglement == "linear":
            # Try a rotation gate change instead
            new_rotations = random.sample(ALLOWED_ROTATIONS, k=2)
            print(f"[LLM-Agent] Trying new rotation gates: {new_rotations}.")

    # ── Reasoning rule 3: good score → explore entanglement upgrade ─────
    elif best_score <= 0.75:
        # Push entanglement up to see if expressivity can lower loss further
        if new_entanglement == "none":
            new_entanglement = "linear"
            print(f"[LLM-Agent] Score good. Upgrading entanglement: none → linear.")
        else:
            print(f"[LLM-Agent] Score good ({best_score:.3f}). Preserving architecture.")

    return {
        "n_qubits": 3,
        "n_layers": new_layers,
        "rotation_gates": new_rotations,
        "entanglement": new_entanglement
    }
