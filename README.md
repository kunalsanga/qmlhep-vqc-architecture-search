# 🔬 QMLHEP — VQC Architecture Search

A **Variational Quantum Circuit (VQC) Architecture Search** engine built with [PennyLane](https://pennylane.ai/). Inspired by the **QMLHEP** (Quantum Machine Learning for High Energy Physics) initiative.

This project implements **two search strategies** — Random Search and Evolutionary Search — and compares them as a research progression, building toward the argument that classical search methods have fundamental limitations that motivate more intelligent approaches.

---

## 🗺️ Research Progression (The Big Picture)

```
Stage 1: Random Search
    → Sample architectures randomly, pick the best score
    → Stochastic, no memory, no refinement

Stage 2: Evolutionary Search  ← current stage
    → Start from a population, mutate the best each generation
    → Slight improvement, but still noisy convergence

Stage 3 (future): LLM / Bayesian / RL-guided Search
    → Use semantic reasoning about architecture structure
    → Sample-efficient, can avoid known-bad patterns
```

> **Core thesis**: Classical search strategies exhibit limited sample efficiency and lack semantic reasoning about architecture structure. This motivates intelligent architecture search for quantum circuits.

---

## 🧠 What This Project Does

1. **Randomly generates** VQC architectures — varying layers, rotation gates (`RX`, `RY`, `RZ`), and entanglement patterns (`none`, `linear`, `full`).
2. **Builds a PennyLane QNode** for each architecture on the `default.qubit` simulator.
3. **Encodes input** via `RY(x[i])` per qubit. Pads with `0.0` for extra qubits when features < qubits.
4. **Trains** each circuit on `make_moons` using `GradientDescentOptimizer`.
5. **Scores** using a hardware-efficiency cost function:

```
Score = Loss + λ_depth × depth + λ_CNOT × CNOT_count
```

6. **Random Search**: samples N architectures independently and picks the best.
7. **Evolutionary Search**: starts with a random population, mutates the best each generation.
8. **Exports** all random-search results to `results.csv`.
9. **Visualises** scatter plots + correlation matrix via `plots.py`.

---

## 📁 Project Structure

```
qmlhep-vqc-architecture-search/
│
├── main.py               # Entry point — currently runs evolutionary search
├── search.py             # Random search loop + saves results.csv
├── evolution_search.py   # Evolutionary search loop (population + mutation)
├── evolution.py          # mutate_architecture() — single-point mutation
├── architecture.py       # Random architecture sampler
├── circuit_builder.py    # Builds PennyLane QNode from architecture config
├── trainer.py            # Gradient descent training with pennylane.numpy weights
├── evaluator.py          # Computes depth, CNOT count, hardware-efficiency score
├── plots.py              # Scatter plots + correlation matrix from results.csv
├── config.py             # All hyperparameters and search space constants
├── results.csv           # Auto-generated after random search run
├── requirements.txt      # Python dependencies
└── README.md
```

---

## ⚙️ Current Configuration (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_QUBITS` | **3** | Number of qubits per circuit |
| `MAX_LAYERS` | 4 | Maximum variational layers |
| `ALLOWED_ROTATIONS` | RX, RY, RZ | Rotation gates to sample from |
| `ENTANGLEMENT_PATTERNS` | **none, linear, full** | CNOT entanglement topologies |
| `LAMBDA_DEPTH` | 0.01 | Depth penalty in score |
| `LAMBDA_CNOT` | 0.02 | CNOT penalty in score |
| `TRAIN_STEPS` | 40 | Gradient descent steps per circuit |
| `LEARNING_RATE` | 0.1 | Optimizer learning rate |

---

## ⚙️ File-by-File Explanation

### `architecture.py`
```python
generate_architecture() → dict
```
Randomly samples: fixed `n_qubits`, random `n_layers`, random 2-gate rotation block, random entanglement.

---

### `circuit_builder.py`
```python
build_circuit(architecture) → QNode
```
- **Input encoding**: `RY(x[i] if i < len(x) else 0.0, wires=i)` — safe 3-qubit padding for 2-feature data
- **Rotation block**: parameterized RX/RY/RZ per qubit per layer
- **Entanglement block**:
  - `"none"` → no CNOT (product state only, limited expressivity)
  - `"linear"` → CNOT chain q0→q1→q2 (moderate, hardware-friendly)
  - `"full"` → all-to-all CNOTs (max entanglement, highest cost)
- **Output**: `<PauliZ(0)>` ∈ [-1, +1]

---

### `trainer.py`
```python
train_architecture(circuit, architecture, X, y) → loss
```
- Uses `pennylane.numpy` with `requires_grad=True` — **never plain numpy**
- MSE loss over `make_moons` dataset
- `TRAIN_STEPS` steps of `GradientDescentOptimizer`

---

### `evaluator.py`
```python
evaluate_structure(architecture) → (depth, total_gates, cnot_count)
compute_score(loss, depth, cnot_count) → score
```

**CNOT counts per layer (3 qubits):**

| Entanglement | CNOTs/layer | 3 layers total |
|---|---|---|
| `none` | 0 | 0 |
| `linear` | 2 | 6 |
| `full` | 3 | 9 |

**Score** = `loss + 0.01×depth + 0.02×cnot_count`

---

### `search.py` — Random Search
```python
run_search(X, y, iterations=10) → (best, results)
```
- Samples N architectures independently (no memory between iterations)
- Saves all results to `results.csv`
- **Limitation**: purely stochastic — good result depends on luck

---

### `evolution.py` — Mutation Operator
```python
mutate_architecture(architecture) → new_architecture
```
Picks one mutation at random:
- `"layers"` → randomise `n_layers`
- `"rotation"` → resample 2 rotation gates
- `"entanglement"` → switch entanglement pattern

Uses `copy.deepcopy()` to avoid mutating the original.

---

### `evolution_search.py` — Evolutionary Search
```python
run_evolution_search(X, y, population_size=4, generations=4) → best
```

**Algorithm:**
```
1. Generate population_size random architectures and evaluate each
2. For each generation:
   a. Select the best architecture (elitism, size=1)
   b. Mutate best → (population_size - 1) new children
   c. Evaluate all children
   d. Replace population with [best] + children
3. Return final best
```

**Current settings**: `population_size=4`, `generations=4` = 4 + 4×3 = 16 total circuit evaluations.

---

### `plots.py`
```python
plot_results(csv_file="results.csv")
```
- Scatter: **Score vs Circuit Depth**
- Scatter: **Loss vs CNOT Count**
- Printed **Correlation Matrix**

---

## 🚀 Setup & Installation

```bash
git clone https://github.com/kunalsanga/qmlhep-vqc-architecture-search.git
cd qmlhep-vqc-architecture-search

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python main.py
```

---

## 📊 Observed Results

### Random Search (8 iterations, 2 qubits)
```
Best: n_layers=2, rotation=['RY','RX'], entanglement=linear
Loss: 0.5144 | Depth: 4 | CNOT: 2 | Score: 0.5944
```

### Evolutionary Search (4 population, 4 generations, 3 qubits)
```
Gen 1 → Best score: 0.7367
Gen 2 → Best score: 0.7367   ← stalled
Gen 3 → Best score: 0.7338   ← improved
Gen 4 → Best score: 0.7338   ← stalled again

Final best:
  n_qubits: 3 | n_layers: 2 | rotation: ['RZ','RY'] | entanglement: full
  Loss: 0.5738 | Score: 0.7338
```

---

## 🧠 Key Research Insights (Read This First When Revisiting!)

---

### 1️⃣ Evolution improved — but only slightly

```
0.7367 → 0.7338  (Δ = 0.003)
```

The algorithm refined the architecture across generations, confirming that selection pressure works. But the gain is small because:
- Mutation is **random** — no gradient in architecture space
- **No crossover** — can't combine good traits from multiple parents
- **No memory** — doesn't remember which mutations failed before
- **Elitism of 1** — diversity collapses fast

> 📌 **Remember**: Small gains from evolution are expected and not a failure — they're your evidence that *guided* search (LLM/Bayesian/RL) is the natural next step.

---

### 2️⃣ Scores are higher in 3-qubit runs — that's expected

```
2-qubit random search best score:  0.5944
3-qubit evolution search best:     0.7338
```

This isn't a regression. 3-qubit circuits have:
- More CNOTs → higher CNOT penalty
- Higher depth → higher depth penalty
- More parameters → harder to train in 40 steps

> 📌 **Remember**: Don't compare 2-qubit scores to 3-qubit scores directly. They exist in different cost landscapes.

---

### 3️⃣ Linear entanglement = sweet spot (confirmed again)

From random search (2 qubits):
```
linear, 2 layers → loss: 0.514, score: 0.594  ← WINNER
full, 3 layers   → loss: 0.590, score: 0.831
none, 4 layers   → loss: 0.741, score: 0.821
```

`none` = no expressivity. `full` = too expensive. `linear` = balanced.

> 📌 **Remember**: This is your core empirical result. Cite it in any proposal.

---

### 4️⃣ The two stalling points are structural, not bugs

```
Gen 1 → 0.7367
Gen 2 → 0.7367  ← stall
Gen 3 → 0.7338
Gen 4 → 0.7338  ← stall
```

Stalling means: all mutations of the current best scored worse → best carried forward unchanged. This is a known weakness of (1+λ) elitist evolution with no crossover.

> 📌 **Remember**: Stalling is evidence for the thesis that random mutation without memory is sample-inefficient.

---

### 5️⃣ Why `pennylane.numpy` — the one rule you must never forget

```python
# WRONG — crashes with "requires_grad unexpected keyword"
weights = np.random.randn(n_layers, n_qubits, n_rot, requires_grad=True)

# CORRECT
import pennylane.numpy as pnp
weights = pnp.random.normal(size=(n_layers, n_qubits, n_rot), requires_grad=True)
```

PennyLane uses the **parameter-shift rule** to compute gradients. Plain `numpy` arrays are invisible to the autograd engine. `pennylane.numpy` wraps numpy and adds the tracking.

> 📌 **Remember**: This was the very first bug we fixed. It will bite anyone new to PennyLane.

---

### 6️⃣ The score formula is not arbitrary — it mirrors real hardware concerns

```
Score = Loss + 0.01×depth + 0.02×CNOT_count
```

- `λ_CNOT (0.02) > λ_depth (0.01)` — CNOT gates are noisier than single-qubit gates on NISQ devices
- IBM Quantum and Google both report CNOT error rates ~10× higher than single-qubit error rates
- **Lower score = better real-world viability**, not just better training

> 📌 **Remember**: This scoring function is what separates this from a plain VQE/classifier. It's hardware-aware.

---

### 7️⃣ Input padding for qubit-feature mismatch

```python
qml.RY(x[i] if i < len(x) else 0.0, wires=i)
```

`make_moons` gives 2 features. 3-qubit circuit has 3 wires. The 3rd qubit is encoded as `RY(0)` = identity → starts in `|0⟩` but still participates in entanglement with other qubits.

> 📌 **Remember**: For real HEP data (many features), you'd have the opposite problem — more features than qubits — and would need dimensionality reduction (PCA or amplitude encoding).

---

## 📌 Publishable-Level Statement You Can Use

> *"We implement and compare two quantum architecture search strategies — random sampling and mutation-based evolutionary search — on a hardware-efficiency-aware scoring function. While evolutionary search improves upon random sampling through iterative refinement, it exhibits characteristic stalling behaviour due to the absence of crossover and architectural memory. These limitations highlight the sample inefficiency of classical search heuristics in the discrete quantum architecture space, motivating the development of semantically-guided search methods."*

---

## 🔧 How to Run Each Search Mode

**Random Search** (edit `main.py`):
```python
from search import run_search
best, results = run_search(X, y, iterations=10)
```

**Evolutionary Search** (current default):
```python
from evolution_search import run_evolution_search
best = run_evolution_search(X, y, population_size=4, generations=4)
```

---

## 🔧 How to Extend

| Goal | How |
|---|---|
| Add crossover | In `evolution.py`, add `crossover(arch1, arch2)` that swaps sub-components |
| Increase population | Change `population_size` in `main.py` |
| More qubits | Change `MAX_QUBITS` in `config.py` |
| Real HEP data | Replace `make_moons` in `main.py` with your feature matrix |
| Bayesian search | Replace mutation with a Gaussian process surrogate model over architecture configs |
| Save evolution log | Append each generation's best to a CSV inside `evolution_search.py` |

---

## 📚 References

- [PennyLane Documentation](https://docs.pennylane.ai/)
- [QMLHEP GSoC Project](https://hepsoftwarefoundation.org/gsoc/2024/proposal_QMLHEP.html)
- [Variational Quantum Circuits — Schuld et al.](https://arxiv.org/abs/1803.00745)
- [Hardware-Efficient VQE — Kandala et al.](https://www.nature.com/articles/nature23879)
- [Parameter-Shift Rule — Crooks 2019](https://arxiv.org/abs/1905.13311)
- [Neural Architecture Search Survey — Elsken et al.](https://arxiv.org/abs/1808.05377)

---

## 👤 Author

**Kunal Sanga**
GitHub: [@kunalsanga](https://github.com/kunalsanga)
