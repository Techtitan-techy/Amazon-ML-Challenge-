import json

new_cell_7 = '''# =========================================================================
# Step 6: Full Test Set Inference with Candidate Generation
# =========================================================================
s1_test_file = TEST_S1
s2_test_file = TEST_S2
s3_test_file = TEST_S3

cand_out = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
match_out = os.path.join(OUTPUT_DIR, "matching_results.tsv")

con = duckdb.connect()
countries = [r[0] for r in con.execute(f"SELECT DISTINCT country FROM read_csv('{s1_test_file}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country IS NOT NULL").fetchall()]
total_s1 = con.execute(f"SELECT count(*) FROM read_csv('{s1_test_file}', delim='\\t', header=True, quote='', all_varchar=True)").fetchone()[0]

print(f"Test Set Countries: {countries} | Total S1: {total_s1:,}")

all_s1_order = []
candidate_pool = []  # (s1_id, tid, prob)
name_lookup = {}     # Stores raw names for championship plaza/suite pruning
total_cands = 0
K_BUDGET = 40
SINGLETON_FLOOR = 0.35
DECISION_THRESHOLD = 0.58

with open(cand_out, 'w', encoding='utf-8') as f_cand:
    f_cand.write("source1_entity_id\\tcandidate_entity_ids\\n")
    
    for country in countries:
        print(f"\\nProcessing Country Partition: {country}...")
        t0 = time.time()
        
        q_df = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('{s1_test_file}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country='{country}'").fetch_df()
        for r in q_df.itertuples(index=False):
            name_lookup[r.entity_id] = str(r.business_name or '')
        q_list = [build_record(r.entity_id, r.business_name, r.business_address, r.country) for r in q_df.itertuples(index=False)]
        del q_df
        
        t_df = con.execute(f"""
            SELECT entity_id, business_name, business_address, country FROM read_csv('{s2_test_file}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country='{country}'
            UNION ALL
            SELECT entity_id, business_name, business_address, country FROM read_csv('{s3_test_file}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country='{country}'
        """).fetch_df()
        for r in t_df.itertuples(index=False):
            name_lookup[r.entity_id] = str(r.business_name or '')
        t_list = [build_record(r.entity_id, r.business_name, r.business_address, r.country) for r in t_df.itertuples(index=False)]
        del t_df
        
        if not t_list:
            for q in q_list:
                all_s1_order.append(q['entity_id'])
                f_cand.write(f"{q['entity_id']}\\t\\n")
            continue
            
        tgt_map = {r['entity_id']: r for r in t_list}
        c_idx = FastCountryIndex(country, t_list)
        print(f"  Built index for {len(q_list):,} queries and {len(t_list):,} targets ({time.time()-t0:.2f}s).")
        
        # Batch retrieval
        BATCH = 5000
        for b in range(0, len(q_list), BATCH):
            b_queries = q_list[b:b+BATCH]
            cands_dict = c_idx.retrieve(b_queries, k=K_BUDGET, batch_size=2000)
            
            pair_meta = []
            pair_inputs = []
            
            for q in b_queries:
                sid = q['entity_id']
                all_s1_order.append(sid)
                cands = cands_dict.get(sid, [])
                cids = [tid for tid, _ in cands]
                f_cand.write(f"{sid}\\t{','.join(cids)}\\n")
                total_cands += len(cids)
                
                for tid, sc in cands:
                    tgt = tgt_map.get(tid)
                    if tgt:
                        pair_meta.append((sid, tid))
                        pair_inputs.append((
                            q['norm_name'], tgt['norm_name'],
                            q['norm_addr'], tgt['norm_addr'],
                            q['postal'], tgt['postal'],
                            q['house_num'], tgt['house_num'],
                            sc
                        ))
                        
            if pair_inputs:
                Xb = compute_features(pair_inputs)
                probs = model.predict_proba(Xb)[:, 1]
                for (sid, tid), p in zip(pair_meta, probs):
                    if p >= SINGLETON_FLOOR:
                        candidate_pool.append((sid, tid, float(p)))
                        
        del c_idx, tgt_map, q_list, t_list
        gc.collect()

print(f"All countries retrieved. Total candidate pairs written: {total_cands:,}")
'''

new_cell_8 = '''# =========================================================================
# Step 7: Global 1-to-1 Deduplication, Championship Pruning & Match Export
# =========================================================================
print("\\nApplying Global 1-to-1 Bipartite Deduplication & Championship Optimization...")
t0 = time.time()

if candidate_pool:
    df_pool = pd.DataFrame(candidate_pool, columns=['s1', 'pool', 'p'])
    del candidate_pool
    gc.collect()
    
    # Every target pool record goes exclusively to the single S1 with the highest probability
    best_p = df_pool.groupby('pool')['p'].transform('max')
    df_resolved = df_pool[df_pool['p'] >= best_p].drop_duplicates(subset=['pool'])
    del df_pool
    gc.collect()
    
    # Group matches by S1 meeting decision threshold
    s1_matches = defaultdict(list)
    for row in df_resolved[df_resolved['p'] >= DECISION_THRESHOLD].itertuples(index=False):
        s1_matches[row.s1].append((row.pool, row.p))
    del df_resolved
    gc.collect()
else:
    s1_matches = {}

# Championship Plaza / Suite False-Positive Pruner (Preserves Indic/French transliterations)
from rapidfuzz.distance import JaroWinkler
LEGAL_RE = re.compile(r'\\b(pvt ltd|private limited|ltd|limited|llc|inc|corp|corporation|co|company|sarl|sas|sa|sci|ets|cie|associates|group)\\b', re.IGNORECASE)

def clean_latin(s):
    if not s:
        return ""
    s = LEGAL_RE.sub("", str(s).lower())
    return re.sub(r'[^a-z0-9 ]', ' ', s).strip()

def cap_matches(raw_matches):
    if len(raw_matches) <= 8:
        return raw_matches
    s2 = [m for m in raw_matches if m.startswith('S2')][:5]
    s3 = [m for m in raw_matches if m.startswith('S3')][:5]
    return (s2 + s3)[:8]

total_matches = 0
pruned_fps = 0
with open(match_out, 'w', encoding='utf-8') as f_match:
    f_match.write("source1_entity_id\\tmatched_entity_ids\\n")
    for sid in all_s1_order:
        cands = s1_matches.get(sid, [])
        if not cands:
            f_match.write(f"{sid}\\t\\n")
        else:
            cands.sort(key=lambda x: x[1], reverse=True)
            
            # Prune blatant Latin plaza/suite collisions sharing an address
            s1_raw = name_lookup.get(sid, '')
            c1 = clean_latin(s1_raw)
            is_s1_ascii = c1.isascii() and len(c1) >= 3
            toks1 = set(c1.split()) if is_s1_ascii else set()
            
            valid_matches = []
            for tid, _ in cands:
                t_raw = name_lookup.get(tid, '')
                c2 = clean_latin(t_raw)
                is_t_ascii = c2.isascii() and len(c2) >= 3
                if is_s1_ascii and is_t_ascii:
                    toks2 = set(c2.split())
                    if len(toks1) > 0 and len(toks2) > 0 and len(toks1 & toks2) == 0:
                        if JaroWinkler.similarity(c1, c2) < 0.45:
                            pruned_fps += 1
                            continue # Prune unrelated business sharing a plaza address
                valid_matches.append(tid)
                
            capped = cap_matches(valid_matches)
            f_match.write(f"{sid}\\t{','.join(capped)}\\n")
            total_matches += len(capped)

del name_lookup
gc.collect()

print(f"\\n=== CHAMPIONSHIP EXPORT COMPLETE ({time.time()-t0:.2f}s) ===")
print(f"  Blatant Plaza False Positives Pruned: {pruned_fps:,}")
print(f"  Total Retained Champion Links:        {total_matches:,} (Avg {total_matches/len(all_s1_order):.2f}/S1)")
'''

with open('Team_Mani_Kaggle_Fast_Pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][7]['source'] = [l + '\n' for l in new_cell_7.strip().split('\n')]
nb['cells'][8]['source'] = [l + '\n' for l in new_cell_8.strip().split('\n')]

with open('Team_Mani_Kaggle_Fast_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Updated Team_Mani_Kaggle_Fast_Pipeline.ipynb successfully!')
