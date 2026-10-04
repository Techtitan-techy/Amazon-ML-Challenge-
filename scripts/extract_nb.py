import json
import sys

with open('kaggle-ce-v5-base2-ipynb.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
with open('extracted_kaggle_notebook.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        cell_type = cell.get('cell_type', '')
        source = "".join(cell.get('source', []))
        out.write(f"\n# {'='*60}\n# CELL {i} [{cell_type}]\n# {'='*60}\n")
        if cell_type == 'markdown':
            out.write('"""\n' + source + '\n"""\n')
        else:
            out.write(source + '\n')

print("Extracted notebook to extracted_kaggle_notebook.py successfully.")
