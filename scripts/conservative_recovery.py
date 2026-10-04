"""
Conservative Post-Processing to Push from 0.988 Toward 0.990
=============================================================
Approach:
1. Start with champion (0.988) as the hard floor
2. Analyze entities that were pruned-to-empty (655 entities)
3. Restore ONLY the "safe" ones: where the target had no collision and is uniquely owned
4. This is conservative enough that precision should not drop
"""
import pandas as pd
import numpy as np

PROJECT_DIR = r"d:\PROJECTS\Amazon ML Challenge"
CHAMPION_FILE = f"{PROJECT_DIR}\\matching_results_champion_0988.tsv"
BASELINE_FILE = f"{PROJECT_DIR}\\matching_results_0979_baseline.tsv"
OUTPUT_FILE = f"{PROJECT_DIR}\\matching_results_conservative_recovery.tsv"

print("Loading files...")
champ = pd.read_csv(CHAMPION_FILE, sep='\t', dtype=str, keep_default_na=False)
base = pd.read_csv(BASELINE_FILE, sep='\t', dtype=str, keep_default_na=False)

print(f"Champion: {len(champ):,} S1 entities")
print(f"Baseline: {len(base):,} S1 entities")

# Build set-based structure
champ_dict = dict(zip(champ['source1_entity_id'], champ['matched_entity_ids']))
base_dict = dict(zip(base['source1_entity_id'], base['matched_entity_ids']))

def parse_ids(s):
    if not s or s == 'nan':
        return set()
    return {x for x in s.split(',') if x}

# Build target -> [s1 entities] map from baseline AND champion
print("Building target->S1 maps...")
base_target_to_s1 = {}
champ_target_to_s1 = {}

for s1_id, ids_str in base_dict.items():
    for t in parse_ids(ids_str):
        if t not in base_target_to_s1:
            base_target_to_s1[t] = []
        base_target_to_s1[t].append(s1_id)

for s1_id, ids_str in champ_dict.items():
    for t in parse_ids(ids_str):
        if t not in champ_target_to_s1:
            champ_target_to_s1[t] = []
        champ_target_to_s1[t].append(s1_id)

# Find entities pruned to empty
pruned_to_empty = []
for s1_id in champ['source1_entity_id']:
    c_ids = parse_ids(champ_dict.get(s1_id, ''))
    b_ids = parse_ids(base_dict.get(s1_id, ''))
    if not c_ids and b_ids:
        pruned_to_empty.append({'s1_id': s1_id, 'base_ids': b_ids})

print(f"\nEntities pruned to empty by champion: {len(pruned_to_empty)}")

# For each pruned entity, check if the targets are uniquely owned
safe_to_restore = []
collision_based = []

for item in pruned_to_empty:
    s1_id = item['s1_id']
    base_ids = item['base_ids']
    
    all_unique = True
    for t in base_ids:
        # Check if this target is uniquely claimed by this S1 in baseline
        claimants = base_target_to_s1.get(t, [])
        if len(claimants) > 1:
            all_unique = False
            break
    
    # Also check: is this target now claimed by another S1 in champion?
    now_claimed_elsewhere = False
    for t in base_ids:
        champ_claimants = champ_target_to_s1.get(t, [])
        if champ_claimants and s1_id not in champ_claimants:
            now_claimed_elsewhere = True
            break
    
    if all_unique and not now_claimed_elsewhere:
        safe_to_restore.append(item)
    else:
        collision_based.append({
            **item,
            'all_unique': all_unique,
            'now_claimed_elsewhere': now_claimed_elsewhere
        })

print(f"  Safe to restore (unique target, not claimed elsewhere): {len(safe_to_restore)}")
print(f"  NOT safe to restore (collision or claimed elsewhere): {len(collision_based)}")

# Show sample of safe-to-restore entities
print("\nSample of safe-to-restore entities:")
for item in safe_to_restore[:5]:
    print(f"  S1: {item['s1_id']}, base_ids: {item['base_ids']}")

# Apply conservative recovery: restore the safe ones to champion
print("\nApplying conservative recovery...")
output_dict = dict(champ_dict)  # Start from champion

restored_count = 0
for item in safe_to_restore:
    s1_id = item['s1_id']
    base_ids = item['base_ids']
    output_dict[s1_id] = ','.join(sorted(base_ids))
    restored_count += 1

print(f"Restored {restored_count} entities to non-empty")

# Build output TSV
output_rows = []
for _, row in champ.iterrows():
    s1_id = row['source1_entity_id']
    matched = output_dict.get(s1_id, '')
    output_rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': matched})

output_df = pd.DataFrame(output_rows)

# Verify structure
total_links_champ = sum(len(parse_ids(v)) for v in champ_dict.values())
total_links_output = sum(len(parse_ids(v)) for v in output_dict.values())
print(f"\nChampion total links: {total_links_champ:,}")
print(f"Output total links: {total_links_output:,}")
print(f"Delta: +{total_links_output - total_links_champ:,} links recovered")

# Check for target uniqueness (ground truth requirement: each target -> exactly 1 S1)
print("\nVerifying 1-to-1 target assignment in output...")
output_target_to_s1 = {}
for s1_id, ids_str in output_dict.items():
    for t in parse_ids(ids_str):
        if t not in output_target_to_s1:
            output_target_to_s1[t] = []
        output_target_to_s1[t].append(s1_id)

multi_claimed = {t: s1s for t, s1s in output_target_to_s1.items() if len(s1s) > 1}
print(f"Targets claimed by >1 S1 in output: {len(multi_claimed)}")
if multi_claimed:
    print("WARNING: Violated 1-to-1 constraint! Applying deduplication...")
    # Keep only the one with fewer total matches (be conservative)
    for t, s1s in multi_claimed.items():
        # Keep the one with the smallest match set (most conservative)
        best_s1 = min(s1s, key=lambda s: len(parse_ids(output_dict.get(s, ''))))
        for s1 in s1s:
            if s1 != best_s1:
                ids = parse_ids(output_dict.get(s1, ''))
                ids.discard(t)
                output_dict[s1] = ','.join(sorted(ids))
    print("Deduplication applied.")

# Write output
print(f"\nWriting output to {OUTPUT_FILE}...")
output_df.to_csv(OUTPUT_FILE, sep='\t', index=False)

# Final stats
output_lengths = [len(parse_ids(v)) for v in output_dict.values()]
print(f"Final output:")
print(f"  S1 with 0 matches: {sum(1 for x in output_lengths if x == 0):,}")
print(f"  S1 with >=1 match: {sum(1 for x in output_lengths if x > 0):,}")
print(f"  Total links: {sum(output_lengths):,}")
print(f"  Champion links: {total_links_champ:,}")
print(f"  Net change: {sum(output_lengths) - total_links_champ:+,}")
print("\nDone! Verify this file before submitting.")
print("Expected: marginal improvement (0.988 -> ~0.9885)")
print("Conservative: will not decrease score significantly")
