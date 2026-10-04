import json

# Read notebook
with open('Team_Mani_Kaggle_Fast_Pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 4 (FastCountryIndex):
new_cell_4 = '''# =========================================================================
# Step 3: Fast Country-Partitioned Inverted Index Retrieval (Ultra-Lean & Safe)
# =========================================================================
class FastCountryIndex:
    def __init__(self, country, target_records):
        self.country = country
        self.target_ids = [r['entity_id'] for r in target_records]
        
        # Exact & Joined-Domain Anchors
        self.exact_name_map = defaultdict(list)
        self.exact_addr_map = defaultdict(list)
        self.exact_joined_map = defaultdict(list)
        
        for idx, r in enumerate(target_records):
            n = r['norm_name']
            a = r['norm_addr']
            j = r['joined_name']
            if n and len(n) >= 3:
                self.exact_name_map[n].append(idx)
            if a and len(a) >= 6:
                self.exact_addr_map[a].append(idx)
            if j and len(j) >= 6:
                self.exact_joined_map[j[:8]].append(idx)
                
        # Word TF-IDF with max_df=0.25 to prune high-frequency noise words ("de", "la", "france", "rue")
        t_texts = [r['norm_full'] for r in target_records]
        self.word_vec = TfidfVectorizer(
            analyzer='word', token_pattern=r'\\b\\w+\\b', min_df=3, max_df=0.25,
            max_features=80000, sublinear_tf=True, dtype=np.float32
        )
        self.t_word_mat = self.word_vec.fit_transform(t_texts)
        del t_texts
        gc.collect()
        
        # Address Char 3-4 Gram TF-IDF
        t_addrs = [r['norm_addr'] for r in target_records]
        self.char_vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 4), min_df=5, max_df=0.25,
            max_features=40000, sublinear_tf=True, dtype=np.float32
        )
        self.t_char_mat = self.char_vec.fit_transform(t_addrs)
        del t_addrs
        gc.collect()

    def retrieve(self, query_records, k=40, batch_size=250):
        q_ids = [r['entity_id'] for r in query_records]
        q_texts = [r['norm_full'] for r in query_records]
        q_addrs = [r['norm_addr'] for r in query_records]
        n_q = len(query_records)
        candidates_by_s1 = defaultdict(dict)
        
        # 1. Exact anchors
        for r in query_records:
            sid = r['entity_id']
            n, a, j = r['norm_name'], r['norm_addr'], r['joined_name']
            if n in self.exact_name_map:
                for tidx in self.exact_name_map[n][:k]:
                    candidates_by_s1[sid][self.target_ids[tidx]] = 1.0
            if a in self.exact_addr_map:
                for tidx in self.exact_addr_map[a][:k]:
                    candidates_by_s1[sid][self.target_ids[tidx]] = max(candidates_by_s1[sid].get(self.target_ids[tidx], 0.0), 0.95)
            if j and len(j) >= 6:
                jk = j[:8]
                if jk in self.exact_joined_map:
                    for tidx in self.exact_joined_map[jk][:k]:
                        candidates_by_s1[sid][self.target_ids[tidx]] = max(candidates_by_s1[sid].get(self.target_ids[tidx], 0.0), 0.90)
                        
        # 2. Batch Word TF-IDF with safe 250 sub-batch to prevent sparse explosion
        for s_idx in range(0, n_q, batch_size):
            e_idx = min(s_idx + batch_size, n_q)
            q_mat = self.word_vec.transform(q_texts[s_idx:e_idx])
            sim = q_mat.dot(self.t_word_mat.T)
            b_sids = q_ids[s_idx:e_idx]
            
            weak_queries = []
            for i, sid in enumerate(b_sids):
                st, en = sim.indptr[i], sim.indptr[i+1]
                if st == en:
                    weak_queries.append((s_idx + i, sid))
                    continue
                t_idx = sim.indices[st:en]
                sc = sim.data[st:en]
                if len(sc) > k:
                    part = np.argpartition(-sc, k)[:k]
                    t_idx, sc = t_idx[part], sc[part]
                order = np.argsort(-sc)
                t_idx, sc = t_idx[order], sc[order]
                if len(sc) == 0 or sc[0] < 0.35:
                    weak_queries.append((s_idx + i, sid))
                for ti, score in zip(t_idx, sc):
                    if score > 0.05:
                        tid = self.target_ids[ti]
                        candidates_by_s1[sid][tid] = max(candidates_by_s1[sid].get(tid, 0.0), float(score))
                        
            # 3. Address char booster for weak queries
            if weak_queries:
                w_indices, w_sids = zip(*weak_queries)
                w_addrs = [q_addrs[idx] for idx in w_indices]
                valid = [(sid, a) for sid, a in zip(w_sids, w_addrs) if len(a) >= 4]
                if valid:
                    vsids, vaddrs = zip(*valid)
                    c_mat = self.char_vec.transform(vaddrs)
                    csim = c_mat.dot(self.t_char_mat.T)
                    for j, sid in enumerate(vsids):
                        cst, cen = csim.indptr[j], csim.indptr[j+1]
                        if cst == cen: continue
                        ct_idx, csc = csim.indices[cst:cen], csim.data[cst:cen]
                        top_n = min(15, len(csc))
                        part = np.argpartition(-csc, top_n)[:top_n]
                        for ti, score in zip(ct_idx[part], csc[part]):
                            if score > 0.12:
                                tid = self.target_ids[ti]
                                candidates_by_s1[sid][tid] = max(candidates_by_s1[sid].get(tid, 0.0), float(score))
                                
        return {sid: sorted(cands.items(), key=lambda x: x[1], reverse=True)[:k] for sid, cands in candidates_by_s1.items()}

print("High-recall inverted index retriever ready (Memory Safe).")
'''

nb['cells'][4]['source'] = [l + '\n' for l in new_cell_4.strip().split('\n')]

# Also ensure Step 6 uses batch_size=250 and BATCH=2000
new_cell_7 = '''# =========================================================================
# Step 6: Full Test Set Inference with Candidate Generation (Memory Safe)
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
target_best = {}  # Compact in-place 1-to-1 best tracker: tid -> (sid, prob). Uses only ~1.2 GB max
total_cands = 0
K_BUDGET = 40
DECISION_THRESHOLD = 0.58

with open(cand_out, 'w', encoding='utf-8') as f_cand:
    f_cand.write("source1_entity_id\\tcandidate_entity_ids\\n")
    
    for country in countries:
        print(f"\\nProcessing Country Partition: {country}...")
        t0 = time.time()
        
        q_df = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('{s1_test_file}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country='{country}'").fetch_df()
        q_list = [build_record(r.entity_id, r.business_name, r.business_address, r.country) for r in q_df.itertuples(index=False)]
        del q_df
        
        t_df = con.execute(f"""
            SELECT entity_id, business_name, business_address, country FROM read_csv('{s2_test_file}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country='{country}'
            UNION ALL
            SELECT entity_id, business_name, business_address, country FROM read_csv('{s3_test_file}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country='{country}'
        """).fetch_df()
        t_list = [build_record(r.entity_id, r.business_name, r.business_address, r.country) for r in t_df.itertuples(index=False)]
        del t_df
        
        if not t_list:
            for q in q_list:
                all_s1_order.append(q['entity_id'])
                f_cand.write(f"{q['entity_id']}\\t\\n")
            continue
            
        tgt_map = {r['entity_id']: r for r in t_list}
        c_idx = FastCountryIndex(country, t_list)
        # Immediately free t_list to release ~4GB RAM
        del t_list
        gc.collect()
        
        print(f"  Built index for {len(q_list):,} queries and {len(tgt_map):,} targets ({time.time()-t0:.2f}s).")
        
        # Batch retrieval with 2000 batch and 250 sub-batch
        BATCH = 2000
        for b in range(0, len(q_list), BATCH):
            b_queries = q_list[b:b+BATCH]
            cands_dict = c_idx.retrieve(b_queries, k=K_BUDGET, batch_size=250)
            
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
                    if p >= DECISION_THRESHOLD:
                        p_val = float(p)
                        cur = target_best.get(tid)
                        if cur is None or p_val > cur[1]:
                            target_best[tid] = (sid, p_val)
                del Xb, probs, pair_inputs, pair_meta
                
        # Completely free country index and queries before next partition
        del c_idx, tgt_map, q_list
        gc.collect()
        print(f"  Country {country} completed! Running matches stored: {len(target_best):,}")

print(f"\\nAll countries retrieved. Total candidate pairs written: {total_cands:,}")
print(f"Total target-exclusive matches above threshold: {len(target_best):,}")
'''

nb['cells'][7]['source'] = [l + '\n' for l in new_cell_7.strip().split('\n')]

with open('Team_Mani_Kaggle_Fast_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Updated Cell 4 and Cell 7 successfully with max_df=0.25 and batch_size=250!")
