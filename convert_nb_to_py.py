import json
import os

notebook_path = r'c:\Users\kunal sanga\Desktop\qmlhep quantum circuit designer\experiments\bottleneck_analysis.ipynb'
script_path = r'c:\Users\kunal sanga\Desktop\qmlhep quantum circuit designer\run_analysis_cells.py'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

script_lines = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        script_lines.extend(cell['source'])
        script_lines.append('\n\n')

with open(script_path, 'w', encoding='utf-8') as f:
    f.writelines(script_lines)

print(f"Generated python script from notebook: {script_path}")
