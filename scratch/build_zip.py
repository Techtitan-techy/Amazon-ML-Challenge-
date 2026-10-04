"""
Build the final submission ZIP for Amazon ML Challenge 2026.

Structure:
  Team_Mani_submission_final.zip
  ├── output/
  │   ├── matching_results.tsv        <- matching_results_champion_0988.tsv
  │   └── candidate_pairs.tsv        <- generated from champion run
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/                   <- copied from Team_Mani_submission/code/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md

NOTE: candidate_pairs.tsv must come from the SAME run that produced matching_results_champion_0988.tsv.
Since it's not stored locally, we generate a fresh one using the local pipeline.
"""

import os, sys, zipfile, shutil, time

BASE = r"d:\PROJECTS\Amazon ML Challenge"
OUT_ZIP = os.path.join(BASE, "Team_Mani_submission_final.zip")

# Source files
MATCHING = os.path.join(BASE, "matching_results_champion_0988.tsv")
CANDS = os.path.join(BASE, "output", "candidate_pairs.tsv")  # might not exist yet
CODE_DIR = os.path.join(BASE, "Team_Mani_submission", "code")
DOC = os.path.join(BASE, "Documentation_template.md")

# Verify key assets
print("=== Asset Check ===")
print(f"matching_results_champion_0988.tsv: {'EXISTS' if os.path.exists(MATCHING) else 'MISSING'} ({os.path.getsize(MATCHING):,} bytes)" if os.path.exists(MATCHING) else "MISSING")
print(f"candidate_pairs.tsv: {'EXISTS' if os.path.exists(CANDS) else 'MISSING - needs generation'}")
print(f"Code dir: {'EXISTS' if os.path.exists(CODE_DIR) else 'MISSING'}")
print(f"Documentation_template.md: {'EXISTS' if os.path.exists(DOC) else 'MISSING'}")

# List code files that will be included
print("\n=== Code files to include ===")
for root, dirs, files in os.walk(CODE_DIR):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
    for f in files:
        full = os.path.join(root, f)
        rel = os.path.relpath(full, CODE_DIR)
        print(f"  code/{rel}: {os.path.getsize(full):,} bytes")
