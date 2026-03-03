import json
import os

notebook_path = r'c:\Users\kunal sanga\Desktop\qmlhep quantum circuit designer\experiments\bottleneck_analysis.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        
        # 1. Imports
        for i, line in enumerate(source):
            if "from llm_search import run_llm_search" in line and "run_rule_based_search" not in line:
                source[i] = "from llm_search import run_llm_search, run_rule_based_search\n"
                
        # 2. Add Rule-Based LLM Strategy handling
        for i, line in enumerate(source):
            if 'elif strategy_name == "LLM":' in line:
                # check if Rule-Based LLM already there
                if not any("Rule-Based LLM" in src_line for src_line in source):
                    new_lines = [
                        '        elif strategy_name == "Rule-Based LLM":\n',
                        '            best_scores, _, _ = strategy_fn(X, y, iterations=iterations)\n'
                    ]
                    source.insert(i, new_lines[0])
                    source.insert(i+1, new_lines[1])
                break
                
        # 3. Add to results computation
        for i, line in enumerate(source):
            if "llm_evals, llm_curves = run_multiple_seeds(run_llm_search," in line:
                if not any("Rule-Based LLM" in src_line for src_line in source):
                    new_lines = [
                        'rb_evals, rb_curves = run_multiple_seeds(run_rule_based_search, "Rule-Based LLM", n_runs=N_RUNS, iterations=MAX_EVALS, threshold=THRESHOLD)\n',
                        'results["Rule-Based LLM"] = {"evals": rb_evals, "curves": rb_curves}\n\n'
                    ]
                    source.insert(i, new_lines[0])
                    source.insert(i+1, new_lines[1])
                break
                
        # 4. Add color for plot 
        for i, line in enumerate(source):
            if "colors =" in line and "'#2ca02c'" in line and "'#d62728'" not in line:
                source[i] = "        colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']\n"

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated.")
