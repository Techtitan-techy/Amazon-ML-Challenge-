"""
Generate candidate_pairs.tsv from the champion matching + baseline.
candidate_pairs must be a SUPERSET of matching_results.
"""
import os

os.makedirs('output', exist_ok=True)

print('Loading baseline candidates...')
baseline = {}
with open('matching_results_0979_baseline.tsv', 'r', encoding='utf-8') as fb:
    next(fb)
    for line in fb:
        parts = line.strip().split('\t')
        sid = parts[0]
        baseline[sid] = parts[1] if len(parts) > 1 else ''

print(f'Loaded {len(baseline):,} baseline rows')

print('Generating candidate_pairs.tsv...')
total = 0
total_cands = 0

with open('matching_results_champion_0988.tsv', 'r', encoding='utf-8') as fin, \
     open('output/candidate_pairs.tsv', 'w', encoding='utf-8') as fout:

    fout.write('source1_entity_id\tcandidate_entity_ids\n')
    next(fin)

    for line in fin:
        parts = line.strip().split('\t')
        sid = parts[0]
        matched = parts[1] if len(parts) > 1 else ''

        champ_set = set(matched.split(',')) if matched else set()
        base_str = baseline.get(sid, '')
        base_set = set(base_str.split(',')) if base_str else set()

        all_cands = champ_set | base_set
        all_cands.discard('')

        cand_str = ','.join(sorted(all_cands))
        fout.write(sid + '\t' + cand_str + '\n')
        total += 1
        total_cands += len(all_cands)

print('Wrote', total, 'rows to output/candidate_pairs.tsv')
print('Total candidates:', total_cands)
print('Size:', os.path.getsize('output/candidate_pairs.tsv'), 'bytes')
