# 🔬 QMLHEP — LLM-Guided VQC Architecture Search

> **A reproducible, feedback-driven framework for Variational Quantum Circuit (VQC) architecture search using real LLM reasoning, built with [PennyLane](https://pennylane.ai/).**

Inspired by the **QMLHEP** (Quantum Machine Learning for High Energy Physics) initiative, this project implements and benchmarks **four search strategies** for discovering optimal quantum circuit architectures — ranging from stochastic baselines to a closed-loop LLM agent that reasons over historical performance feedback via a local [Ollama](https://ollama.com/) instance.

**Core Thesis:** *Classical search strategies exhibit limited sample efficiency and lack semantic reasoning about architecture structure. An LLM-guided agent conditions proposals on historical performance metrics, enabling more principled and sample-efficient exploration of the quantum architecture space.*

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py (Orchestrator)                      │
│   Runs all search strategies sequentially, prints summary, plots    │
└────────┬──────────────────┬──────────────────┬──────────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌────────────────┐ ┌──────────────────────────────┐
│  search.py      │ │ evolution_     │ │  llm_search.py               │
│  Random Search  │ │ search.py      │ │  LLM-Guided + Rule-Based     │
│  (Baseline)     │ │ Evolutionary   │ │  Closed-loop feedback search  │
└────────┬────────┘ └───────┬────────┘ └─────────────┬────────────────┘
         │                  │                        │
         │          ┌───────┴─────┐          ┌───────┴────────────────┐
         │          │ evolution.py│          │ llm_agent.py            │
         │          │ Mutation Op │          │ Ollama LLM ↔ JSON parse │
         │          └─────────────┘          │ + Rule-based fallback   │
         │                                   └────────────────────────┘
         │                  │                        │
         ▼                  ▼                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Shared Evaluation Pipeline                        │
│                                                                      │
│  architecture.py ──► circuit_builder.py ──► trainer.py ──► evaluator │
│  (Random sampler)    (PennyLane QNode)     (Grad Descent)  (Score)   │
│                                                                      │
│  config.py ──► All hyperparameters, search space bounds, penalties   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Project Structure

```
qmlhep-vqc-architecture-search/
│
├── main.py                # Entry point — runs all strategies + comparison plot
│
├── llm_agent.py           # 🧠 Real LLM agent (Ollama) + rule-based fallback
├── llm_search.py          # 🔁 Closed-loop LLM search + reasoning trace logger
│
├── evolution_search.py    # Evolutionary search: 1+λ elitist strategy
├── evolution.py           # mutate_architecture() — single-point mutation
│
├── search.py              # Random search: independent samples baseline
├── architecture.py        # Random architecture sampler
│
├── circuit_builder.py     # Builds PennyLane QNode from architecture config
├── trainer.py             # Trains circuits via parameter-shift gradients
├── evaluator.py           # Hardware-aware scoring (depth + CNOT penalties)
│
├── comparison_plot.py     # Convergence comparison figure (all strategies)
├── plots.py               # Scatter plots from results.csv
├── run_analysis_cells.py  # Multi-seed bottleneck analysis (script version)
│
├── config.py              # All hyperparameters and search space constants
├── requirements.txt       # Python dependencies
│
├── experiments/
│   ├── bottleneck_analysis.ipynb   # Jupyter notebook: multi-seed benchmarking
│   ├── llm_reasoning_trace.json    # Full LLM reasoning trace dataset
│   ├── results.csv                 # Experiment results
│   └── run_eval.py                 # Standalone evaluation script
│
├── results.csv            # Auto-generated after random search
├── comparison_plot.png    # Auto-generated convergence comparison figure
└── README.md
```

---

## ⚙️ Configuration (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_QUBITS` | 3 | Number of qubits per circuit |
| `MAX_LAYERS` | 4 | Maximum variational layers |
| `ALLOWED_ROTATIONS` | RX, RY, RZ | Single-qubit rotation gate set |
| `ENTANGLEMENT_PATTERNS` | none, linear, full | CNOT entanglement topologies |
| `LAMBDA_DEPTH` | 0.01 | Depth penalty coefficient |
| `LAMBDA_CNOT` | 0.02 | CNOT penalty coefficient |
| `TRAIN_STEPS` | 40 | Gradient descent steps per circuit |
| `LEARNING_RATE` | 0.1 | Optimizer learning rate |

---

## 🤖 LLM Integration via Ollama

The framework integrates a **real local LLM** for architecture proposal through [Ollama](https://ollama.com/). This replaces the earlier rule-based heuristic agent with genuine language model reasoning.

### How It Works

1. **Structured Prompt Construction** — `llm_agent.py` builds a detailed prompt containing:
   - The last 3 architecture evaluations (architecture config, loss, depth, CNOT count, final score)
   - Explicit constraints (max qubits, max layers, allowed gates, entanglement options)
   - Strict instruction to output **only valid JSON**, no markdown or explanation

2. **Local Ollama Query** — The prompt is sent to `http://localhost:11434/api/generate` using `urllib` (no external SDK dependencies). Default model: **`qwen2.5`**.

3. **Safe JSON Parsing** — Response is parsed via regex + `json.loads` with:
   - Bounded value clamping (qubits ∈ [1, MAX_QUBITS], layers ∈ [1, MAX_LAYERS])
   - Rotation gate validation against `ALLOWED_ROTATIONS`
   - Entanglement pattern validation against `ENTANGLEMENT_PATTERNS`
   - **No `eval()` usage** — fully safe execution

4. **Deterministic Fallback** — If the LLM returns invalid output or the connection fails, the system falls back to a random architecture generator. The search loop never crashes.

5. **Reasoning Trace Logging** — Every LLM call logs:
   - Full prompt sent
   - Raw LLM output text
   - Parsed architecture dictionary
   - Final hardware-aware score
   
   All traces are saved to `experiments/llm_reasoning_trace.json`.

### Key Design Properties

- **Fully offline** — runs entirely on localhost via Ollama
- **No API keys required** — no OpenAI, no cloud dependencies
- **Reproducible** — deterministic fallback + logged traces
- **Modular** — swap the model by changing `DEFAULT_MODEL` in `llm_agent.py`

---

## 🔬 Search Strategies

### Stage 1: Random Search (Baseline)
```python
from search import run_search
best_scores, results = run_search(X, y, iterations=8)
```
- N independent architecture samples with no memory between evaluations
- Establishes the stochastic baseline for comparison

### Stage 2: Evolutionary Search (1+λ Elitist)
```python
from evolution_search import run_evolution_search
evo_scores, best = run_evolution_search(X, y, population_size=4, generations=4)
```
- Initialises a random population, evaluates each
- Each generation: selects the single best (elitism), mutates to fill next generation
- Single-point mutation: randomise layers, rotation gates, or entanglement pattern

### Stage 3: Rule-Based Agent (Heuristic Baseline)
```python
from llm_search import run_rule_based_search
rb_scores, best, history = run_rule_based_search(X, y, iterations=6)
```
- Deterministic heuristics that simulate feedback-conditioned reasoning
- Used as the control agent for comparison against the real LLM

### Stage 4: LLM-Guided Search (Real LLM)
```python
from llm_search import run_llm_search
llm_scores, best, history = run_llm_search(X, y, iterations=6)
```
- Closed feedback loop: LLM receives full history → proposes architecture → evaluates → feeds result back
- Every LLM response is logged as a reasoning trace

---

## 🏗️ Hardware-Aware Multi-Objective Scoring

The scoring function reflects the constraints of real NISQ quantum hardware:

```
Score = Loss + λ_depth × Depth + λ_CNOT × CNOT_count
```

Where:
- **Loss** = MSE training loss of the parameterised circuit
- **Depth** = `n_layers × 2` (approximate circuit depth)
- **CNOT count** = entanglement-dependent two-qubit gate count

**Why `λ_CNOT (0.02) > λ_depth (0.01)`?**

On IBM Quantum and Google hardware, CNOT gates exhibit ~10× higher error rates than single-qubit rotations. The asymmetric penalty directly encodes this hardware reality into the search objective. A circuit that achieves low loss but requires many CNOT gates will score poorly — matching real-world device constraints.

| Entanglement | CNOTs per layer (3 qubits) | Trade-off |
|---|---|---|
| `none` | 0 | High loss, low penalty |
| `linear` | 2 | Balanced — typically optimal |
| `full` | 3 | Low loss, high CNOT penalty |

---

## 📊 Experimental Validation

### Latest Run Results (3 qubits, `make_moons` dataset)

| Strategy | Best Score | Total Evals | Key Observation |
|---|---|---|---|
| Random Search | 0.7115 | 8 | No refinement, pure stochastic sampling |
| Evolutionary | 0.6995 | 16 | Stalls when all mutations score worse than parent |
| **LLM-Guided** | **0.7037** | **6** | Achieves competitive score with fewest evaluations |

### Convergence Behaviour

- **Random search** plateaus quickly — no feedback mechanism to refine proposals
- **Evolutionary search** stalls structurally due to lack of crossover and diversity collapse under elitism
- **LLM-guided search** demonstrates sample-efficient convergence by conditioning proposals on the full evaluation history

### Multi-Seed Benchmarking

The `experiments/bottleneck_analysis.ipynb` notebook and `run_analysis_cells.py` script run all four strategies across multiple random seeds to compute:
- Mean evaluations to reach a score threshold
- Standard deviation and variance of convergence
- Average convergence curves with confidence bands
- Search space combinatorial explosion analysis

### Reasoning Trace Dataset

Every LLM call is logged in `experiments/llm_reasoning_trace.json` with the following schema:
```json
{
  "iteration": 2,
  "prompt": "You are an expert quantum circuit architect...",
  "raw_output": "{\"qubits\": 3, \"layers\": 3, ...}",
  "parsed_architecture": {
    "n_qubits": 3,
    "n_layers": 3,
    "rotation_gates": ["RX", "RY", "RZ"],
    "entanglement": "linear"
  },
  "final_score": 0.7037
}
```

This provides a complete audit trail of the LLM agent's reasoning for analysis, reproducibility, and interpretability.

---

## 🚀 Setup & Installation

### 1. Clone and Set Up Python Environment

```bash
git clone https://github.com/kunalsanga/qmlhep-vqc-architecture-search.git
cd qmlhep-vqc-architecture-search

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. Install Ollama (Required for LLM-Guided Search)

```bash
# Download and install from https://ollama.com/download

# Pull the default model
ollama pull qwen2.5

# Start the Ollama server (runs on http://localhost:11434)
ollama serve
```

### 3. Run the Full Pipeline

```bash
python main.py
```

This runs all three stages sequentially (Random → Evolutionary → LLM-Guided), prints a results summary, and saves `comparison_plot.png`.

### 4. Run Multi-Seed Bottleneck Analysis

```bash
python run_analysis_cells.py
```

Runs all four strategies (Random, Evolutionary, Rule-Based, LLM) across multiple seeds, generates summary statistics, and saves convergence plots.

---

## 🔧 Module Reference

### `circuit_builder.py`
Builds a PennyLane `@qml.qnode` from an architecture dictionary:
- **Input encoding**: `RY(x[i])` per qubit — zero-pads when features < qubits
- **Rotation block**: Parameterised RX/RY/RZ gates per qubit per layer
- **Entanglement block**: CNOT topology per layer (`none` / `linear` / `full`)
- **Output**: `⟨PauliZ(0)⟩` expectation value ∈ [-1, +1]

### `trainer.py`
- Weights initialised with `pennylane.numpy` (`requires_grad=True`) — never plain numpy
- MSE loss: `Σ (prediction - label)² / N`
- Optimised via `qml.GradientDescentOptimizer` using the **parameter-shift rule**

### `evaluator.py`
```python
evaluate_structure(architecture) → (depth, total_gates, cnot_count)
compute_score(loss, depth, cnot_count) → float
```

### `llm_agent.py`
```python
propose_architecture_with_trace(history) → (architecture, prompt, raw_response)
llm_generate_architecture(history) → architecture  # backward-compatible alias
rule_based_generate_architecture(history) → architecture  # heuristic baseline
```

---

## ⚠️ Current Limitations

- **Small search space**: Fixed at 3 qubits with a maximum of 4 layers. The combinatorial space (~324 modular architectures) is tractable for random search, limiting the advantage demonstrated by LLM guidance.
- **Synthetic dataset**: Evaluated on `make_moons` (2D, 20 samples) rather than real HEP data.
- **Single objective**: While hardware-aware, the scoring does not incorporate noise simulation, gate fidelity variation, or qubit connectivity constraints.
- **LLM consistency**: The local LLM (Qwen 2.5) occasionally proposes architectures that are structurally similar across iterations. Temperature and sampling parameters are not yet tuned.
- **No crossover in evolutionary search**: The 1+λ strategy uses only mutation, limiting its ability to combine good substructures from different architectures.
- **Limited statistical runs**: Due to the computational cost of quantum circuit training, multi-seed benchmarks are capped at 5 seeds × 20 iterations per strategy.

---

## 🔮 Future Work

- **Scale to larger circuits**: Increase qubit count to 6–10 to demonstrate LLM advantage in exponentially growing search spaces (3^Q rotation combinations)
- **Real HEP data integration**: Replace `make_moons` with actual particle physics features (e.g., jet tagging, event classification)
- **Noise-aware scoring**: Integrate Qiskit Aer noise models to simulate realistic device error rates
- **Bayesian surrogate model**: Use a Gaussian Process over architecture configurations to further improve sample efficiency
- **Crossover operators**: Add recombination to evolutionary search for fairer comparison
- **Multi-objective optimisation**: Pareto-front analysis across loss, depth, and CNOT count
- **LLM fine-tuning**: Train a domain-specific model on the reasoning trace dataset for specialised quantum architecture reasoning
- **Real quantum hardware**: Deploy discovered architectures on IBM Quantum or Google devices via `qml.device("qiskit.ibmq", ...)`

---

## 📚 References

- [PennyLane Documentation](https://docs.pennylane.ai/)
- [QMLHEP GSoC Project](https://hepsoftwarefoundation.org/gsoc/2024/proposal_QMLHEP.html)
- [Variational Quantum Circuits — Schuld et al.](https://arxiv.org/abs/1803.00745)
- [Hardware-Efficient VQE — Kandala et al.](https://www.nature.com/articles/nature23879)
- [Parameter-Shift Rule — Crooks 2019](https://arxiv.org/abs/1905.13311)
- [LLM Agents for Scientific Discovery — Wang et al.](https://arxiv.org/abs/2304.05332)
- [Neural Architecture Search Survey — Elsken et al.](https://arxiv.org/abs/1808.05377)
- [Ollama](https://ollama.com/) — Local LLM inference

---

## 👤 Author

**Kunal Sanga**  
GitHub: [@kunalsanga](https://github.com/kunalsanga)
