import json

nb = json.load(open('Team_Mani_Kaggle_Fast_Pipeline.ipynb', encoding='utf-8'))
with open('scratch/fast_pipeline_dump.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        out.write(f"\n{'='*70}\n=== CELL {i} ({c['cell_type']}) ===\n{'='*70}\n")
        out.write(src)
        out.write("\n")
print("Dumped", len(nb['cells']), "cells")
