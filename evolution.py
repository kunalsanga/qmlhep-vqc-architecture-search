import random
import copy
from config import MAX_LAYERS, ALLOWED_ROTATIONS, ENTANGLEMENT_PATTERNS


def mutate_architecture(architecture):

    new_arch = copy.deepcopy(architecture)

    mutation_type = random.choice(["layers", "rotation", "entanglement"])

    if mutation_type == "layers":
        new_arch["n_layers"] = random.randint(1, MAX_LAYERS)

    elif mutation_type == "rotation":
        new_arch["rotation_gates"] = random.sample(ALLOWED_ROTATIONS, k=2)

    elif mutation_type == "entanglement":
        new_arch["entanglement"] = random.choice(ENTANGLEMENT_PATTERNS)

    return new_arch