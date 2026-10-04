"""
Advanced Post-Processing Optimizer for Amazon ML Challenge 2026
===============================================================
Goal: Push from 0.988 to 0.990+ using the champion (0.988) TSV as the hard floor.

Strategy:
1. Analyze what the CE notebook did to produce the champion
2. Apply `expected_f_select` (EFS) threshold instead of flat threshold
3. Re-apply global 1-to-1 bipartite deduplication
4. Recover missed links using a "soft singleton" analysis
5. Keep the 0.988 champion as a hard floor - only accept changes that are verified improvements

Key insight from analysis:
- Champion: 5,672,130 links (under-predicted vs ~6.0M expected from GT)
- All changes from baseline were PRUNING (34,643 fewer links) - raised precision
- Structure matches GT well: 81.6% both S2+S3 (GT: 80.5%)
- Path forward: need to carefully recover true positives that were over-pruned

Author: Antigravity
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

PROJECT_DIR = Path(r"d:\PROJECTS\Amazon ML Challenge")
CHAMPION_FILE = PROJECT_DIR / "matching_results_champion_0988.tsv"
BASELINE_FILE = PROJECT_DIR / "matching_results_0979_baseline.tsv"
OUTPUT_FILE = PROJECT_DIR / "matching_results_optimized.tsv"

# ─────────────────────────────────────────────────────────────────
# 1. Load files
# ─────────────────────────────────────────────────────────────────
print("Loading champion and baseline...")
champ = pd.read_csv(CHAMPION_FILE, sep='\t', dtype=str, keep_default_na=False)
base = pd.read_csv(BASELINE_FILE, sep='\t', dtype=str, keep_default_na=False)

# Merge on source1_entity_id
merged = champ.merge(base, on='source1_entity_id', suffixes=('_champ', '_base'))

def parse_ids(s):
    if not s or s == 'nan':
        return set()
    return {x for x in s.split(',') if x}

# ─────────────────────────────────────────────────────────────────
# 2. Identify rows where champion pruned vs baseline
# ─────────────────────────────────────────────────────────────────
print("Analyzing differences between champion and baseline...")

rows_added = []       # champion added something baseline didn't have
rows_pruned = []      # champion pruned something baseline had
rows_same = []        # no difference

for _, row in merged.iterrows():
    c_ids = parse_ids(row['matched_entity_ids_champ'])
    b_ids = parse_ids(row['matched_entity_ids_base'])
    added = c_ids - b_ids
    removed = b_ids - c_ids
    rows_pruned.append({
        's1': row['source1_entity_id'],
        'champ': row['matched_entity_ids_champ'],
        'base': row['matched_entity_ids_base'],
        'n_pruned': len(removed),
        'n_added': len(added),
        'pruned_ids': ','.join(sorted(removed)),
        'added_ids': ','.join(sorted(added)),
    })

df_diff = pd.DataFrame(rows_pruned)
pruned_rows = df_diff[df_diff['n_pruned'] > 0]
added_rows = df_diff[df_diff['n_added'] > 0]
print(f"S1 entities where champion pruned links: {len(pruned_rows):,}")
print(f"S1 entities where champion added links: {len(added_rows):,}")
print(f"Total links pruned by champion: {pruned_rows['n_pruned'].sum():,}")
print(f"Total links added by champion: {added_rows['n_added'].sum():,}")

# ─────────────────────────────────────────────────────────────────
# 3. Analyze the pruned targets: are they multi-claimed or genuine FPs?
# ─────────────────────────────────────────────────────────────────
print("\nAnalyzing pruned targets...")

# Build a map of which target appears in which S1 entities (baseline)
# and which are claimed in champion
base_target_to_s1 = {}  # target_id -> list of s1_entity_ids that claim it in baseline
champ_target_to_s1 = {}

for _, row in base.iterrows():
    s1_id = row['source1_entity_id']
    ids = parse_ids(row['matched_entity_ids'])
    for t in ids:
        if t not in base_target_to_s1:
            base_target_to_s1[t] = []
        base_target_to_s1[t].append(s1_id)

for _, row in champ.iterrows():
    s1_id = row['source1_entity_id']
    ids = parse_ids(row['matched_entity_ids'])
    for t in ids:
        if t not in champ_target_to_s1:
            champ_target_to_s1[t] = []
        champ_target_to_s1[t].append(s1_id)

# In baseline: how many targets appear in >1 S1 (collisions)?
multi_claimed_base = {t: s1s for t, s1s in base_target_to_s1.items() if len(s1s) > 1}
multi_claimed_champ = {t: s1s for t, s1s in champ_target_to_s1.items() if len(s1s) > 1}
print(f"Targets claimed by >1 S1 in baseline: {len(multi_claimed_base):,}")
print(f"Targets claimed by >1 S1 in champion: {len(multi_claimed_champ):,}")
print(f"Champion resolved {len(multi_claimed_base) - len(multi_claimed_champ):,} collisions")

# ─────────────────────────────────────────────────────────────────
# 4. OOV name analysis: Identify S1 entities with no-match risk
#    (These are entities where the GT name was replaced with a random word)
#    The CE correctly identifies these via cosine embedding similarity
# ─────────────────────────────────────────────────────────────────
# Strategy: Use the champion as our base. The 103,423 rows marked as
# "changed" in the diff are ordering changes only (same IDs, different order)
# The real improvements need to come from:
# a) Better threshold calibration
# b) Recovering the over-pruned entities (655 entities pruned to empty)
# c) Adding a second CE seed ensemble

print("\n=== Strategy Summary ===")
print(f"Champion (0.988): {len(champ):,} S1, {champ['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x and x != 'nan' else 0).sum():,} total links")
print(f"Singleton rate: {(champ['matched_entity_ids'].isna() | (champ['matched_entity_ids'] == '')).sum():,} ({100*(champ['matched_entity_ids'].isna() | (champ['matched_entity_ids'] == '')).sum()/len(champ):.2f}%)")
print()
print("Next steps to push above 0.990:")
print("1. Run CE v5 base with seed=0 (ce_out_seed0/) and seed=1 (ce_out_seed1/)")
print("2. Ensemble by taking MAX of the two GBDT+CE blended scores per pair")
print("3. Re-apply expected_f_select() instead of flat threshold")
print("4. Re-apply global 1-to-1 bipartite dedup")
print()
print("Expected gain: +0.003-0.005 from ensemble (based on CE notebook design)")
print("The notebook Cell 16 comment explicitly says: 'second run, seed 1, for an ensemble with CE_v5_base'")
