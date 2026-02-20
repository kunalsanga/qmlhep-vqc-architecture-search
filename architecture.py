import random
from config import MAX_QUBITS, MAX_LAYERS, ALLOWED_ROTATIONS, ENTANGLEMENT_PATTERNS


def generate_architecture():
    architecture = {
        "n_qubits": MAX_QUBITS,
        "n_layers": random.randint(1, MAX_LAYERS),
        "rotation_gates": random.sample(ALLOWED_ROTATIONS, k=2),
        "entanglement": random.choice(ENTANGLEMENT_PATTERNS)
    }

    return architecture