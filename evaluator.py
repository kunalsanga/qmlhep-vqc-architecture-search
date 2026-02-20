from config import LAMBDA_DEPTH, LAMBDA_CNOT


def evaluate_structure(architecture):

    n_layers = architecture["n_layers"]
    n_qubits = architecture["n_qubits"]
    entanglement = architecture["entanglement"]

    # Rotation gate count
    rotation_count = n_layers * n_qubits * len(architecture["rotation_gates"])

    # CNOT count
    if entanglement == "none":
        cnot_per_layer = 0

    elif entanglement == "linear":
        cnot_per_layer = n_qubits - 1

    elif entanglement == "full":
        cnot_per_layer = n_qubits * (n_qubits - 1) // 2

    cnot_count = cnot_per_layer * n_layers

    total_gates = rotation_count + cnot_count

    # Approximate depth (rotation + entanglement per layer)
    depth = n_layers * 2

    return depth, total_gates, cnot_count


def compute_score(loss, depth, cnot_count):
    return loss + LAMBDA_DEPTH * depth + LAMBDA_CNOT * cnot_count