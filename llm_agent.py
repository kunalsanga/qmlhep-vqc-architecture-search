import json
import random
import urllib.request
import urllib.error
import re
from config import ALLOWED_ROTATIONS, ENTANGLEMENT_PATTERNS, MAX_LAYERS, MAX_QUBITS
from evaluator import evaluate_structure

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5"

PROMPT_TEMPLATE = """You are an expert quantum circuit architect.
Your goal is to propose a new Variational Quantum Circuit (VQC) architecture that minimizes a hardware-aware score.
The score is defined as: loss + penalty(depth) + penalty(CNOT count). Lower is better.

Here is a summary of the latest architectures evaluated (up to 3):
{history_text}

Based on this history, propose a new architecture.
Constraints:
- "qubits" must be an integer (max {max_qubits}).
- "layers" must be an integer (max {max_layers}).
- "rotation_gates" must be a list of strings from: {allowed_rotations}.
- "entanglement" must be one of: {allowed_entanglement}.

You must respond strictly with valid JSON. Do not include any markdown formatting, code blocks, or explanations. Only output the JSON object.

Format:
{{
  "qubits": int,
  "layers": int,
  "rotation_gates": ["RX", "RY", "RZ"],
  "entanglement": "none" | "linear" | "full"
}}
"""

def generate_random_architecture():
    """Fallback architecture generator if LLM fails."""
    return {
        "n_qubits": min(3, MAX_QUBITS),
        "n_layers": random.randint(1, min(2, MAX_LAYERS)),
        "rotation_gates": random.sample(ALLOWED_ROTATIONS, k=random.randint(1, len(ALLOWED_ROTATIONS))),
        "entanglement": random.choice(ENTANGLEMENT_PATTERNS)
    }

def build_prompt(history):
    """Builds the instruction prompt given architecture history."""
    history_text = ""
    if not history:
        history_text = "No history available. This is the first architecture."
    else:
        last_3 = history[-3:]
        for i, record in enumerate(last_3, 1):
            arch = record["architecture"]
            loss = record["loss"]
            score = record["score"]
            depth, _, cnot = evaluate_structure(arch)
            
            history_text += f"---\nAttempt {i}:\n"
            history_text += f"Architecture: {json.dumps(arch)}\n"
            history_text += f"Loss: {loss:.4f}\n"
            history_text += f"Depth: {depth}\n"
            history_text += f"CNOTs: {cnot}\n"
            history_text += f"Final Score: {score:.4f}\n"
            
    return PROMPT_TEMPLATE.format(
        history_text=history_text,
        max_qubits=MAX_QUBITS,
        max_layers=MAX_LAYERS,
        allowed_rotations=json.dumps(ALLOWED_ROTATIONS),
        allowed_entanglement=json.dumps(ENTANGLEMENT_PATTERNS)
    )

def query_ollama(prompt, model=DEFAULT_MODEL):
    """Sends the prompt to the local Ollama instance strictly configured."""
    data = {
        "prompt": prompt,
        "model": model,
        "stream": False
    }
    req = urllib.request.Request(
        OLLAMA_URL, 
        data=json.dumps(data).encode("utf-8"), 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        print(f"[Ollama] Error communicating with Ollama: {e}")
        return ""

def parse_llm_response(response_text):
    """Safely extracts and parses JSON out of the LLM response."""
    try:
        # Regex to safely find the first JSON-like object
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            text = match.group(0)
        else:
            text = response_text
            
        data = json.loads(text)
        
        # Clamp bounded values
        qubits = max(1, min(int(data.get("qubits", 3)), MAX_QUBITS))
        layers = max(1, min(int(data.get("layers", 2)), MAX_LAYERS))
        
        # Validate rotation gates
        rotations = data.get("rotation_gates", ["RY", "RZ"])
        if not isinstance(rotations, list):
            rotations = ["RY", "RZ"]
        
        rotations = [str(r).upper() for r in rotations if str(r).upper() in ALLOWED_ROTATIONS]
        if not rotations: # Must not be empty
            rotations = ["RY", "RZ"]
            
        # Validate entanglement
        entanglement = str(data.get("entanglement", "linear")).lower()
        if entanglement not in ENTANGLEMENT_PATTERNS:
            entanglement = "linear"
            
        return {
            "n_qubits": qubits,
            "n_layers": layers,
            "rotation_gates": rotations,
            "entanglement": entanglement
        }
    except Exception as e:
        print(f"[LLM Parser] Failed to parse JSON: {e}")
        return None

def propose_architecture_with_trace(history):
    """Proposes a new valid architecture using the LLM model logic and returns tracing details."""
    prompt = build_prompt(history)
    print(f"[LLM] Querying Ollama with {len(history)} history records...")
    
    response_text = query_ollama(prompt)
    if not response_text:
        print("[LLM] Empty response or connection error. Falling back to random architecture.")
        return generate_random_architecture(), prompt, ""
    
    arch = parse_llm_response(response_text)
    if arch is None:
        print("[LLM] Format error. Falling back to random architecture.")
        return generate_random_architecture(), prompt, response_text
        
    return arch, prompt, response_text

def propose_architecture(history):
    """Proposes a new valid architecture using the LLM model logic."""
    arch, _, _ = propose_architecture_with_trace(history)
    return arch

def llm_generate_architecture(history):
    """
    Simulates an LLM agent proposing a VQC architecture. 
    Alias to keep backward compatibility with modules previously calling llm_generate_architecture.
    """
    return propose_architecture(history)

def rule_based_generate_architecture(history):
    """The original heuristic-based logic used prior to the real LLM."""
    if not history:
        return {
            "n_qubits": 3,
            "n_layers": 2,
            "rotation_gates": ["RY", "RZ"],
            "entanglement": "linear"
        }

    best = min(history, key=lambda x: x["score"])
    best_arch = best["architecture"]
    best_loss = best["loss"]
    best_score = best["score"]

    new_layers = best_arch["n_layers"]
    new_entanglement = best_arch["entanglement"]
    new_rotations = best_arch["rotation_gates"]

    if best_loss > 0.65:
        new_layers = min(new_layers + 1, MAX_LAYERS)
    elif best_loss <= 0.65 and best_score > 0.75:
        if new_entanglement == "full":
            new_entanglement = "linear"
        elif new_entanglement == "linear":
            new_rotations = random.sample(ALLOWED_ROTATIONS, k=min(2, len(ALLOWED_ROTATIONS)))
    elif best_score <= 0.75:
        if new_entanglement == "none":
            new_entanglement = "linear"

    return {
        "n_qubits": 3,
        "n_layers": new_layers,
        "rotation_gates": new_rotations,
        "entanglement": new_entanglement
    }
