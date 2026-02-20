import pennylane as qml
import numpy as np


def build_circuit(architecture):

    n_qubits = architecture["n_qubits"]
    n_layers = architecture["n_layers"]
    rotation_gates = architecture["rotation_gates"]
    entanglement = architecture["entanglement"]

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, x):

        # Encode input
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)

        # Variational layers
        for layer in range(n_layers):

            # Rotation block
            for q in range(n_qubits):
                for i, gate in enumerate(rotation_gates):

                    if gate == "RX":
                        qml.RX(weights[layer, q, i], wires=q)

                    elif gate == "RY":
                        qml.RY(weights[layer, q, i], wires=q)

                    elif gate == "RZ":
                        qml.RZ(weights[layer, q, i], wires=q)

            # Entanglement block
            if entanglement == "linear":
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            elif entanglement == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])

        return qml.expval(qml.PauliZ(0))

    return circuit