import pennylane as qml
import numpy as np
import pennylane.numpy as pnp
from config import TRAIN_STEPS, LEARNING_RATE


def train_architecture(circuit, architecture, X, y):

    n_layers = architecture["n_layers"]
    n_qubits = architecture["n_qubits"]
    n_rot = len(architecture["rotation_gates"])

    # Initialize weights
    weights = pnp.random.normal(size=(n_layers, n_qubits, n_rot), requires_grad=True)

    opt = qml.GradientDescentOptimizer(LEARNING_RATE)

    def loss_fn(weights):
        loss = 0
        for i in range(len(X)):
            prediction = circuit(weights, X[i])
            loss += (prediction - y[i]) ** 2
        return loss / len(X)

    for step in range(TRAIN_STEPS):
        weights = opt.step(loss_fn, weights)

    final_loss = loss_fn(weights)

    return final_loss