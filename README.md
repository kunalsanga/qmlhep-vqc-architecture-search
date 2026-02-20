# 🔬 QMLHEP — VQC Architecture Search

A lightweight **Variational Quantum Circuit (VQC) Architecture Search** engine built with [PennyLane](https://pennylane.ai/). Inspired by the goals of the **QMLHEP** (Quantum Machine Learning for High Energy Physics) initiative.

The system randomly samples quantum circuit architectures, trains each with gradient descent, evaluates a hardware-efficiency score, selects the **best-performing, lowest-complexity circuit** automatically, saves all results to CSV, and generates scatter plots + a correlation matrix for analysis.

---

## 🧠 What This Project Does

1. **Randomly generates** VQC architectures (varying layers, rotation gates, entanglement patterns).
2. **Builds a PennyLane QNode** for each architecture using the `default.qubit` simulator.
3. **Trains** each circuit on a synthetic binary classification dataset (`make_moons`) using `GradientDescentOptimizer`.
4. **Scores** each circuit using a hardware-efficiency-aware cost function:

```
Score = Loss + λ_depth × depth + λ_CNOT × CNOT_count
```

5. **Selects** the best architecture (lowest score) across all search iterations.
6. **Exports** all results to `results.csv` for inspection and reproducibility.
7. **Visualises** results with scatter plots (Score vs Depth, Loss vs CNOT) and prints a correlation matrix.

---

## 📁 Project Structure

```
qmlhep-vqc-architecture-search/
│
├── main.py            # Entry point — runs the search, prints best, then plots
├── search.py          # Orchestrates the full search loop + saves results.csv
├── architecture.py    # Randomly generates a circuit architecture config
├── circuit_builder.py # Builds a PennyLane QNode from an architecture config
├── trainer.py         # Trains a circuit using PennyLane's gradient descent optimizer
├── evaluator.py       # Computes circuit depth, CNOT count, and hardware-efficiency score
├── plots.py           # Loads results.csv and generates analysis plots
├── config.py          # All hyperparameters and search space constants
├── results.csv        # Auto-generated after each run (all iteration results)
├── requirements.txt   # Python dependencies
└── README.md
```

---

## ⚙️ File-by-File Explanation

### `config.py`
Central configuration. Defines the search space and training hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_QUBITS` | 2 | Number of qubits in every circuit |
| `MAX_LAYERS` | 4 | Maximum variational layers to sample |
| `ALLOWED_ROTATIONS` | RX, RY, RZ | Rotation gates to sample from |
| `ENTANGLEMENT_PATTERNS` | linear, full | CNOT entanglement topologies |
| `LAMBDA_DEPTH` | 0.01 | Penalty weight for circuit depth in score |
| `LAMBDA_CNOT` | 0.02 | Penalty weight for CNOT count in score |
| `TRAIN_STEPS` | 40 | Gradient descent steps per circuit |
| `LEARNING_RATE` | 0.1 | Optimizer learning rate |

---

### `architecture.py`
```python
generate_architecture() → dict
```
Randomly samples a circuit config:
- Fixed `n_qubits = MAX_QUBITS`
- Random `n_layers` ∈ [1, MAX_LAYERS]
- Random 2-gate rotation block from `ALLOWED_ROTATIONS`
- Random entanglement pattern from `ENTANGLEMENT_PATTERNS`

---

### `circuit_builder.py`
```python
build_circuit(architecture) → QNode
```
Constructs a PennyLane `@qml.qnode` with:
- **Input encoding**: `RY(x[i])` on each qubit
- **Variational layers**: parameterized RX/RY/RZ rotations per qubit
- **Entanglement**:
  - `linear` → CNOT chain: q0→q1→q2...
  - `full` → all-to-all CNOT pairs
- **Output**: `<PauliZ(0)>` expectation value (range: [-1, +1])

---

### `trainer.py`
```python
train_architecture(circuit, architecture, X, y) → loss
```
- Initializes weights using `pennylane.numpy` with `requires_grad=True` (required for PennyLane's autograd to differentiate via the parameter-shift rule)
- Defines MSE loss: `Σ (prediction - y_i)² / N`
- Runs `TRAIN_STEPS` steps of `qml.GradientDescentOptimizer`
- Returns final training loss

> **Note**: Plain `numpy` cannot be used here — `pennylane.numpy` wraps it with autograd support so the optimizer can compute parameter-shift gradients.

---

### `evaluator.py`
```python
evaluate_structure(architecture) → (depth, total_gates, cnot_count)
compute_score(loss, depth, cnot_count) → score
```
Computes hardware-efficiency metrics:
- **Depth**: `n_layers × 2` (rotation block + entanglement block per layer)
- **CNOT count**:
  - `linear`: `(n_qubits - 1) × n_layers`
  - `full`: `n_qubits(n_qubits-1)/2 × n_layers`
- **Score** (lower = better): `loss + 0.01×depth + 0.02×cnot_count`

---

### `search.py`
```python
run_search(X, y, iterations=10) → (best, results)
```
Runs the full architecture search loop:
- For each iteration: generate → build → train → evaluate → score
- After all iterations, **saves results to `results.csv`** with columns: `loss`, `depth`, `cnot`, `score`, `architecture`
- Returns the architecture with the lowest score

---

### `plots.py`
```python
plot_results(csv_file="results.csv")
```
Loads `results.csv` with pandas and produces:
- **Scatter plot 1** — Score vs Circuit Depth
- **Scatter plot 2** — Loss vs CNOT Count
- **Correlation matrix** (printed to console) — shows relationships between `loss`, `depth`, `cnot`, and `score`

Sample correlation output from a real run:
```
Correlation Matrix:
           loss     depth      cnot     score
loss   1.000000 -0.783597 -0.783597  0.972961
depth -0.783597  1.000000  1.000000 -0.618916
cnot  -0.783597  1.000000  1.000000 -0.618916
score  0.972961 -0.618916 -0.618916  1.000000
```
> Interpretation: deeper circuits (more layers/CNOTs) tend to achieve **lower loss** (better fit), but are penalised by the hardware-efficiency score. The score is dominated by loss (`0.97` correlation).

---

### `main.py`
Entry point. Loads `make_moons` dataset, scales features to [0, π], converts labels to {-1, +1}, then:
1. Calls `run_search()` for 8 iterations
2. Prints the best architecture found
3. Calls `plot_results()` for visual analysis

---

### `results.csv` (auto-generated)
Written after each run. Columns:

| Column | Description |
|--------|-------------|
| `loss` | Final training MSE |
| `depth` | Approximate circuit depth |
| `cnot` | Total CNOT gate count |
| `score` | Hardware-efficiency score |
| `architecture` | Full architecture config dict |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/kunalsanga/qmlhep-vqc-architecture-search.git
cd qmlhep-vqc-architecture-search
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Architecture Search
```bash
python main.py
```

This will:
- Run 8 search iterations
- Print per-iteration stats to console
- Save all results to `results.csv`
- Display scatter plots
- Print correlation matrix

---

## 📊 Sample Output

```
===== Iteration 1 =====
Architecture: {'n_qubits': 2, 'n_layers': 1, 'rotation_gates': ['RZ', 'RY'], 'entanglement': 'linear'}
Loss: 0.7539
Depth: 2
CNOT: 1
Score: 0.7939

===== Iteration 3 =====
Architecture: {'n_qubits': 2, 'n_layers': 2, 'rotation_gates': ['RY', 'RX'], 'entanglement': 'linear'}
Loss: 0.5144
Depth: 4
CNOT: 2
Score: 0.5944

...

===== BEST ARCHITECTURE FOUND =====
{'architecture': {'n_qubits': 2, 'n_layers': 2, 'rotation_gates': ['RY', 'RX'], 'entanglement': 'linear'},
 'loss': 0.5144, 'depth': 4, 'cnot': 2, 'score': 0.5944}

Correlation Matrix:
           loss     depth      cnot     score
loss   1.000000 -0.783597 -0.783597  0.972961
...
```

---

## 🔭 Key Concepts

| Concept | Description |
|---------|-------------|
| **VQC** | Variational Quantum Circuit — parameterized quantum circuit trained via classical optimization |
| **QNode** | PennyLane's quantum node that links a quantum function to a device |
| **Parameter-Shift Rule** | Gradient computation method for quantum circuits (used internally by PennyLane) |
| **NAS** | Neural Architecture Search — here adapted to quantum circuits |
| **Hardware Efficiency** | Prefer circuits with fewer CNOTs and shallower depth (less noise-sensitive) |
| **make_moons** | Sklearn toy dataset: 2-class, non-linearly separable — good for testing quantum classifiers |
| **Correlation Matrix** | Shows how `loss`, `depth`, `cnot`, and `score` relate across all search iterations |

---

## 🔧 How to Extend

- **More qubits**: increase `MAX_QUBITS` in `config.py`
- **More rotation gates**: add entries to `ALLOWED_ROTATIONS` in `config.py`
- **Real HEP dataset**: replace `make_moons` in `main.py` with your feature vectors
- **More iterations**: change `iterations=8` in `main.py`
- **Better plots**: enhance `plots.py` — add histograms, Pareto frontier, or colour-coded scatter by entanglement type
- **Evolutionary search**: replace random sampling in `architecture.py` with a mutation/crossover loop

---

## 📚 References

- [PennyLane Documentation](https://docs.pennylane.ai/)
- [QMLHEP GSoC Project](https://hepsoftwarefoundation.org/gsoc/2024/proposal_QMLHEP.html)
- [Variational Quantum Circuits — Schuld et al.](https://arxiv.org/abs/1803.00745)
- [Hardware-Efficient VQE — Kandala et al.](https://www.nature.com/articles/nature23879)

---

## 👤 Author

**Kunal Sanga**  
GitHub: [@kunalsanga](https://github.com/kunalsanga)
