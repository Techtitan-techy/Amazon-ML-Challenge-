import json

with open('Team_Mani_Kaggle_Fast_Pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 4 (FastCountryIndex)
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
                
        # Word TF-IDF: max_df=0.25 prunes dense stop words ("france", "de", "la", "rue", "pvt", "ltd")
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
                        
        # 2. Batch Word TF-IDF with safe 250 sub-batch to prevent sparse buffer explosion
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

# 2. Update Cell 6 (Step 5: Smart Model Check)
new_cell_6 = '''# =========================================================================
# Step 5: Option B — Fresh High-Capacity Training / Instant Model Loader
# =========================================================================
saved_model_path = os.path.join(OUTPUT_DIR, "xgboost_matcher.joblib")
has_train_data = bool(TRAIN_GT and TRAIN_S1 and TRAIN_S2 and TRAIN_S3)

if os.path.exists(saved_model_path):
    print(f"Loading already-trained model from {saved_model_path} in 0.1s...")
    model = joblib.load(saved_model_path)
    print("Model loaded successfully! Ready for full test inference.")
elif has_train_data:
    print("\\n=======================================================")
    print("COMMENCING OPTION B: TRAINING FRESH MODEL FROM DATASET")
    print("=======================================================")
    con = duckdb.connect()
    s1_p = TRAIN_S1
    s2_p = TRAIN_S2
    s3_p = TRAIN_S3
    gt_p = TRAIN_GT
    
    SAMPLE_N = 150000
    t0 = time.time()
    
    # 1. Sample True Positives from Ground Truth
    print("1. Sampling 150,000 Verified Ground Truth Pairs across S2 & S3...")
    pos_df = con.execute(f"""
        WITH gt AS (
            SELECT source1_entity_id as s1_id, unnest(string_split(matched_entity_ids, ',')) as tid
            FROM read_csv('{gt_p}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
            USING SAMPLE {SAMPLE_N} (reservoir, 42)
        ),
        tgts AS (
            SELECT entity_id as tid, business_name, business_address, country FROM read_csv('{s2_p}', delim='\\t', header=True, quote='', all_varchar=True)
            UNION ALL
            SELECT entity_id as tid, business_name, business_address, country FROM read_csv('{s3_p}', delim='\\t', header=True, quote='', all_varchar=True)
        )
        SELECT s.entity_id as s1_id, s.business_name as s1_name, s.business_address as s1_addr, s.country as s1_country,
               t.tid, t.business_name as t_name, t.business_address as t_addr, 1.0 as label
        FROM gt g
        JOIN read_csv('{s1_p}', delim='\\t', header=True, quote='', all_varchar=True) s ON g.s1_id = s.entity_id
        JOIN tgts t ON g.tid = t.tid
    """).fetch_df()
    
    # 2. Sample Hard Negatives from BOTH S2 and S3 (Ultra-Fast Equi-Hash Join)
    print("2. Mining 150,000 Hard Negatives (Prefix/Sub-Address Collisions)...")
    neg_df = con.execute(f"""
        WITH sampled_s1 AS (
            SELECT entity_id, business_name, business_address, country, substring(lower(business_name), 1, 4) as pref
            FROM read_csv('{s1_p}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE business_name IS NOT NULL AND length(trim(business_name)) >= 4
            USING SAMPLE 50000 (reservoir, 99)
        ),
        tgts AS (
            SELECT entity_id as tid, business_name, business_address, country, substring(lower(business_name), 1, 4) as pref
            FROM read_csv('{s2_p}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE business_name IS NOT NULL AND length(trim(business_name)) >= 4
            USING SAMPLE 150000 (reservoir, 99)
            UNION ALL
            SELECT entity_id as tid, business_name, business_address, country, substring(lower(business_name), 1, 4) as pref
            FROM read_csv('{s3_p}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE business_name IS NOT NULL AND length(trim(business_name)) >= 4
            USING SAMPLE 150000 (reservoir, 99)
        ),
        raw_negs AS (
            SELECT s.entity_id as s1_id, s.business_name as s1_name, s.business_address as s1_addr, s.country as s1_country,
                   t.tid, t.business_name as t_name, t.business_address as t_addr, 0.0 as label
            FROM sampled_s1 s
            JOIN tgts t ON s.country = t.country AND s.pref = t.pref AND s.entity_id != t.tid
            USING SAMPLE 200000 (reservoir, 42)
        ),
        sampled_gt AS (
            SELECT source1_entity_id as s1_id, unnest(string_split(matched_entity_ids, ',')) as tid
            FROM read_csv('{gt_p}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE source1_entity_id IN (SELECT entity_id FROM sampled_s1)
        )
        SELECT n.*
        FROM raw_negs n
        LEFT JOIN sampled_gt g ON n.s1_id = g.s1_id AND n.tid = g.tid
        WHERE g.tid IS NULL
        USING SAMPLE {len(pos_df)} (reservoir, 42)
    """).fetch_df()
    
    train_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"  Total Pairs: {len(train_df):,} ({len(pos_df):,} Positives, {len(neg_df):,} Negatives) in {time.time()-t0:.2f}s.")
    
    # 3. Compute SIMD Features
    t0 = time.time()
    print("3. Extracting 18 RapidFuzz SIMD Features...")
    train_pairs = []
    for r in train_df.itertuples(index=False):
        n1 = normalize_name(r.s1_name, r.s1_country)
        n2 = normalize_name(r.t_name, r.s1_country)
        a1 = normalize_address(r.s1_addr, r.s1_country)
        a2 = normalize_address(r.t_addr, r.s1_country)
        p1 = extract_postal(r.s1_addr, r.s1_country)
        p2 = extract_postal(r.t_addr, r.s1_country)
        h1 = extract_house_num(r.s1_addr)
        h2 = extract_house_num(r.t_addr)
        ret_s = len(set(n1.split()) & set(n2.split())) / max(1, len(set(n1.split()) | set(n2.split())))
        train_pairs.append((n1, n2, a1, a2, p1, p2, h1, h2, ret_s))
        
    X_all = compute_features(train_pairs)
    y_all = train_df['label'].values.astype(np.float32)
    print(f"  Extracted features in {time.time()-t0:.2f}s.")
    
    # 4. Train/Validation Split (85% Train / 15% Validation)
    split_idx = int(0.85 * len(X_all))
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_val, y_val = X_all[split_idx:], y_all[split_idx:]
    
    # 5. Fit GPU/CPU XGBoost Matcher
    use_gpu = xgb.config.get_config().get('use_cuda', False) or os.system("nvidia-smi >nul 2>&1" if os.name=='nt' else "nvidia-smi >/dev/null 2>&1") == 0
    tree_method = 'hist'
    device = 'cuda' if use_gpu else 'cpu'
    
    print(f"4. Fitting XGBoost Classifier on {device.upper()} (Hist Gradient Boosting)...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method=tree_method,
        device=device,
        eval_metric='logloss',
        random_state=42
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    # 6. Validation Score Report
    val_preds = model.predict_proba(X_val)[:, 1]
    val_roc = roc_auc_score(y_val, val_preds)
    val_pr = average_precision_score(y_val, val_preds)
    
    print("\\n" + "=" * 55)
    print(f"MODEL TRAINING FIT REPORT ({device.upper()})")
    print("=" * 55)
    print(f"  Fit Time:                {fit_time:.2f} seconds")
    print(f"  Train Samples:           {len(X_train):,}")
    print(f"  Validation Samples:      {len(X_val):,}")
    print(f"  Validation ROC-AUC:      {val_roc:.5f}")
    print(f"  Validation PR-AUC:       {val_pr:.5f}")
    print("=" * 55)
    
    # Save model artifact
    joblib.dump(model, saved_model_path)
else:
    raise FileNotFoundError("Train dataset was not found and no pre-trained model was uploaded.")
'''

nb['cells'][6]['source'] = [l + '\n' for l in new_cell_6.strip().split('\n')]

# 3. Update Cell 7 (Step 6: Test Inference)
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
target_best = {}  # In-place 1-to-1 best tracker: tid -> (sid, prob). Compact memory (~1.2 GB max)
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
        
        # Batch retrieval with 2000 batch and safe 250 sub-batch
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

# 4. Update Cell 8 (Step 7: 0.990 Championship Pruning & Match Export)
new_cell_8 = '''# =========================================================================
# Step 7: Global Deduplication, Championship Pruning & Match Export (0.990+)
# =========================================================================
print("\\nApplying Championship Optimization & Cardinality Caps...")
t0 = time.time()

# 1. Invert target_best map to s1_matches
s1_matches = defaultdict(list)
for tid, (sid, p) in target_best.items():
    s1_matches[sid].append((tid, p))
del target_best
gc.collect()

# 2. Fast DuckDB Load of entity names for Latin Plaza/Suite FP filter
print("Loading raw names for plaza false-positive pruning...")
s1_names = dict(con.execute(f"SELECT entity_id, business_name FROM read_csv('{s1_test_file}', delim='\\t', header=True, quote='', all_varchar=True)").fetchall())
t_names = dict(con.execute(f"""
    SELECT entity_id, business_name FROM read_csv('{s2_test_file}', delim='\\t', header=True, quote='', all_varchar=True)
    UNION ALL
    SELECT entity_id, business_name FROM read_csv('{s3_test_file}', delim='\\t', header=True, quote='', all_varchar=True)
""").fetchall())

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
            s1_raw = s1_names.get(sid, '')
            c1 = clean_latin(s1_raw)
            is_s1_ascii = c1.isascii() and len(c1) >= 3
            toks1 = set(c1.split()) if is_s1_ascii else set()
            
            valid_matches = []
            for tid, _ in cands:
                t_raw = t_names.get(tid, '')
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

del s1_names, t_names, s1_matches
gc.collect()

print(f"\\n=== CHAMPIONSHIP EXPORT COMPLETE ({time.time()-t0:.2f}s) ===")
print(f"  Blatant Plaza False Positives Pruned: {pruned_fps:,}")
print(f"  Total Retained Champion Links:        {total_matches:,} (Avg {total_matches/len(all_s1_order):.2f}/S1)")
'''

nb['cells'][8]['source'] = [l + '\n' for l in new_cell_8.strip().split('\n')]

# Save notebook
with open('Team_Mani_Kaggle_Fast_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Full 0.990 pipeline updated in Team_Mani_Kaggle_Fast_Pipeline.ipynb successfully!")
