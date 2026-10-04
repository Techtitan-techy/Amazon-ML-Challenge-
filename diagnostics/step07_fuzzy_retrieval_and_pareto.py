import os
import sys
import time
import json
import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, REPORTS_DIR
from normalization import transliterated, extract_postal, extract_house_num

def batch_top_k(query_matrix, index_matrix, k=20, batch_size=500):
    """
    Computes top-k nearest neighbors using sparse cosine similarity in memory-efficient batches.
    Returns: list of (query_idx, target_idx, similarity)
    """
    n_queries = query_matrix.shape[0]
    results = []
    
    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        batch_queries = query_matrix[start_idx:end_idx]
        
        sim_matrix = batch_queries.dot(index_matrix.T).tocsr()
        
        for q_offset in range(end_idx - start_idx):
            q_idx = start_idx + q_offset
            r_start = sim_matrix.indptr[q_offset]
            r_end = sim_matrix.indptr[q_offset + 1]
            if r_start == r_end:
                continue
            
            target_indices = sim_matrix.indices[r_start:r_end]
            scores = sim_matrix.data[r_start:r_end]
            
            if len(scores) > k:
                # Top k selection using argpartition
                top_k_part = np.argpartition(-scores, k)[:k]
                top_target_idx = target_indices[top_k_part]
                top_scores = scores[top_k_part]
                # Sort descending
                sort_order = np.argsort(-top_scores)
                top_target_idx = top_target_idx[sort_order]
                top_scores = top_scores[sort_order]
            else:
                sort_order = np.argsort(-scores)
                top_target_idx = target_indices[sort_order]
                top_scores = scores[sort_order]
                
            for t_idx, s in zip(top_target_idx, top_scores):
                if s > 0.05: # Minimal similarity threshold
                    results.append((q_idx, t_idx, float(s)))
                    
    return results

def run_step07():
    print("=" * 70)
    print("STEPS 1 - 4: FUZZY RETRIEVAL, HYBRID UNION, MISSED-GT & PARETO CURVE")
    print("=" * 70)
    
    con = duckdb.connect()
    
    # -------------------------------------------------------------
    # 1. Prepare Evaluation Cohort & Target Pool
    # -------------------------------------------------------------
    print("\n[1/5] Sampling evaluation cohort of S1 records...")
    t0 = time.time()
    
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE eval_s1 AS
        WITH s1_labeled AS (
            SELECT 
                s.entity_id as s1_id,
                s.business_name as s1_name,
                s.business_address as s1_addr,
                s.country,
                g.matched_entity_ids
            FROM read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True) s
            JOIN read_csv('{TRAIN_GT}', delim='\\t', header=True, quote='', all_varchar=True) g 
              ON s.entity_id = g.source1_entity_id
        )
        SELECT * FROM s1_labeled
        USING SAMPLE 20000 (reservoir, 42)
    """)
    
    eval_s1_df = con.execute("SELECT * FROM eval_s1").fetch_df()
    print(f"  Loaded {len(eval_s1_df):,} S1 evaluation records ({time.time()-t0:.2f}s).")
    
    eval_gt_pairs = con.execute("""
        SELECT 
            s1_id,
            unnest(string_split(matched_entity_ids, ',')) as target_id
        FROM eval_s1
        WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
    """).fetch_df()
    
    total_eval_gt = len(eval_gt_pairs)
    print(f"  Evaluation cohort contains {total_eval_gt:,} true Ground Truth pairs.")
    gt_pair_set = set(zip(eval_gt_pairs['s1_id'], eval_gt_pairs['target_id']))
    all_true_target_ids = list(eval_gt_pairs['target_id'].unique())
    
    print("\n[2/5] Building target index (True targets + 250,000 distractors)...")
    t0 = time.time()
    
    # Register true target IDs
    con.register('df_true_target_ids', pd.DataFrame({'entity_id': all_true_target_ids}))
    
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE all_targets_pool AS
        SELECT entity_id, business_name, business_address, country, 'S2' as src
        FROM read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True)
        UNION ALL
        SELECT entity_id, business_name, business_address, country, 'S3' as src
        FROM read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True)
    """)
    
    con.execute("""
        CREATE OR REPLACE TEMP TABLE target_corpus AS
        -- All true targets for evaluation cohort
        SELECT t.*
        FROM all_targets_pool t
        JOIN df_true_target_ids g ON t.entity_id = g.entity_id
        UNION
        -- Plus 250,000 random distractors
        SELECT * FROM (
            SELECT * FROM all_targets_pool
            USING SAMPLE 250000 (reservoir, 42)
        )
    """)
    
    target_corpus_df = con.execute("SELECT * FROM target_corpus").fetch_df()
    print(f"  Target corpus size: {len(target_corpus_df):,} records ({time.time()-t0:.2f}s).")
    
    # Pre-normalize queries and targets
    print("\n[3/5] Normalizing strings with transliteration...")
    t0 = time.time()
    eval_s1_df['norm_name'] = eval_s1_df['s1_name'].apply(transliterated)
    eval_s1_df['norm_addr'] = eval_s1_df['s1_addr'].apply(transliterated)
    eval_s1_df['norm_full'] = eval_s1_df['norm_name'] + " " + eval_s1_df['norm_addr']
    eval_s1_df['postal'] = eval_s1_df.apply(lambda r: extract_postal(r['s1_addr'], r['country']), axis=1)
    
    target_corpus_df['norm_name'] = target_corpus_df['business_name'].apply(transliterated)
    target_corpus_df['norm_addr'] = target_corpus_df['business_address'].apply(transliterated)
    target_corpus_df['norm_full'] = target_corpus_df['norm_name'] + " " + target_corpus_df['norm_addr']
    target_corpus_df['postal'] = target_corpus_df.apply(lambda r: extract_postal(r['business_address'], r['country']), axis=1)
    print(f"  Normalization finished in {time.time()-t0:.2f}s.")
    
    # -------------------------------------------------------------
    # 2. Benchmark Fuzzy Retrieval Channels (Char n-gram vs Token TF-IDF)
    # -------------------------------------------------------------
    print("\n[4/5] Running Vectorized Fuzzy Retrieval Channels by Country...")
    
    channels = {}
    channel_recalls = {}
    channel_candidates = {}
    
    # We evaluate for each country partition
    for country in ['US', 'India']:
        print(f"\n--- Country Partition: {country} ---")
        q_sub = eval_s1_df[eval_s1_df['country'] == country].reset_index(drop=True)
        t_sub = target_corpus_df[target_corpus_df['country'] == country].reset_index(drop=True)
        
        q_ids = q_sub['s1_id'].values
        t_ids = t_sub['entity_id'].values
        
        sub_gt_set = {pair for pair in gt_pair_set if pair[0] in set(q_ids)}
        print(f"  Queries: {len(q_sub):,} | Targets: {len(t_sub):,} | True Pairs: {len(sub_gt_set):,}")
        
        # 1. Exact normalized matches
        t_name_map = defaultdict(list)
        t_addr_map = defaultdict(list)
        for idx, row in t_sub.iterrows():
            if row['norm_name'] and len(row['norm_name']) > 2:
                t_name_map[row['norm_name']].append(row['entity_id'])
            if row['norm_addr'] and len(row['norm_addr']) > 5:
                t_addr_map[row['norm_addr']].append(row['entity_id'])
                
        exact_name_pairs = set()
        exact_addr_pairs = set()
        for idx, row in q_sub.iterrows():
            s1 = row['s1_id']
            if row['norm_name'] in t_name_map:
                for tid in t_name_map[row['norm_name']]:
                    exact_name_pairs.add((s1, tid))
            if row['norm_addr'] in t_addr_map:
                for tid in t_addr_map[row['norm_addr']]:
                    exact_addr_pairs.add((s1, tid))
                    
        # 2. Name Character 3-4 Gram TF-IDF
        print("  Vectorizing Name Character 3-4 Grams...")
        t0 = time.time()
        name_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=75000)
        t_name_mat = name_vec.fit_transform(t_sub['norm_name'])
        q_name_mat = name_vec.transform(q_sub['norm_name'])
        print(f"    Index shape: {t_name_mat.shape} ({time.time()-t0:.2f}s). Retrieving top-20...")
        
        t0 = time.time()
        name_results = batch_top_k(q_name_mat, t_name_mat, k=20, batch_size=1000)
        name_pairs = {(q_ids[q_i], t_ids[t_i]) for q_i, t_i, _ in name_results}
        print(f"    Name Fuzzy retrieval: {len(name_pairs):,} candidate pairs in {time.time()-t0:.2f}s.")
        
        # 3. Address Character 3-4 Gram TF-IDF
        print("  Vectorizing Address Character 3-4 Grams...")
        t0 = time.time()
        addr_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=75000)
        t_addr_mat = addr_vec.fit_transform(t_sub['norm_addr'])
        q_addr_mat = addr_vec.transform(q_sub['norm_addr'])
        print(f"    Index shape: {t_addr_mat.shape} ({time.time()-t0:.2f}s). Retrieving top-20...")
        
        t0 = time.time()
        addr_results = batch_top_k(q_addr_mat, t_addr_mat, k=20, batch_size=1000)
        addr_pairs = {(q_ids[q_i], t_ids[t_i]) for q_i, t_i, _ in addr_results}
        print(f"    Address Fuzzy retrieval: {len(addr_pairs):,} candidate pairs in {time.time()-t0:.2f}s.")
        
        # 4. Word Token TF-IDF (Full Name + Address)
        print("  Vectorizing Word Token TF-IDF (Name + Address)...")
        t0 = time.time()
        word_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\b\w+\b', min_df=2, max_features=100000)
        t_word_mat = word_vec.fit_transform(t_sub['norm_full'])
        q_word_mat = word_vec.transform(q_sub['norm_full'])
        word_results = batch_top_k(q_word_mat, t_word_mat, k=20, batch_size=1000)
        word_pairs = {(q_ids[q_i], t_ids[t_i]) for q_i, t_i, _ in word_results}
        print(f"    Word Fuzzy retrieval: {len(word_pairs):,} candidate pairs in {time.time()-t0:.2f}s.")
        
        # Store for country
        channels[country] = {
            'sub_gt_set': sub_gt_set,
            'n_queries': len(q_sub),
            'exact_name': exact_name_pairs,
            'exact_addr': exact_addr_pairs,
            'name_char_tfidf': name_pairs,
            'addr_char_tfidf': addr_pairs,
            'word_tfidf': word_pairs,
            'name_ranked_results': name_results,
            'addr_ranked_results': addr_results,
            'word_ranked_results': word_results,
            'q_ids': q_ids,
            't_ids': t_ids
        }

    # -------------------------------------------------------------
    # 3. Channel Recall & Hybrid Candidate Union
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING RETRIEVAL CHANNELS ACROSS ALL EVALUATION ENTITIES")
    print("=" * 60)
    
    total_gt_all = sum(len(c['sub_gt_set']) for c in channels.values())
    total_q_all = sum(c['n_queries'] for c in channels.values())
    
    channel_eval = []
    for ch_name in ['exact_name', 'exact_addr', 'name_char_tfidf', 'addr_char_tfidf', 'word_tfidf']:
        total_hits = 0
        total_cands = 0
        for cntry, data in channels.items():
            pairs = data[ch_name]
            total_hits += len(pairs.intersection(data['sub_gt_set']))
            total_cands += len(pairs)
        rec = round(float(total_hits / total_gt_all), 4)
        avg_cand = round(float(total_cands / total_q_all), 2)
        channel_eval.append({
            'channel': ch_name,
            'recall': rec,
            'hits': total_hits,
            'total_candidates': total_cands,
            'avg_cand_per_s1': avg_cand
        })
        print(f"  {ch_name:18s} | Recall: {rec*100:>6.2f}% | Avg Cand/S1: {avg_cand:>6.2f} | Total Hits: {total_hits:>6,d}/{total_gt_all:,d}")

    # HYBRID UNION
    print("\n--- Computing Hybrid Union ---")
    all_hybrid_pairs = set()
    total_hybrid_hits = 0
    all_missed_gt = []
    
    for cntry, data in channels.items():
        hybrid_c = data['exact_name'].union(
            data['exact_addr']
        ).union(
            data['name_char_tfidf']
        ).union(
            data['addr_char_tfidf']
        ).union(
            data['word_tfidf']
        )
        
        hits = len(hybrid_c.intersection(data['sub_gt_set']))
        total_hybrid_hits += hits
        all_hybrid_pairs.update(hybrid_c)
        
        # Identify missed ground truth
        missed = data['sub_gt_set'] - hybrid_c
        for s1, tgt in missed:
            all_missed_gt.append({'s1_id': s1, 'target_id': tgt, 'country': cntry})
            
    hybrid_recall = round(float(total_hybrid_hits / total_gt_all), 4)
    avg_hybrid_cand = round(float(len(all_hybrid_pairs) / total_q_all), 2)
    print(f"\n  >>> HYBRID UNION RECALL: {hybrid_recall*100:.2f}% <<<")
    print(f"  >>> Average Candidates per S1: {avg_hybrid_cand} (Total candidate pairs: {len(all_hybrid_pairs):,})")
    print(f"  >>> Reduction Ratio: {1.0 - (len(all_hybrid_pairs) / (total_q_all * 300000)):.8f}")
    
    # Save hybrid recall report
    df_hybrid_report = pd.DataFrame(channel_eval)
    df_hybrid_report.loc[len(df_hybrid_report)] = {
        'channel': 'HYBRID_UNION_ALL',
        'recall': hybrid_recall,
        'hits': total_hybrid_hits,
        'total_candidates': len(all_hybrid_pairs),
        'avg_cand_per_s1': avg_hybrid_cand
    }
    hybrid_report_path = os.path.join(REPORTS_DIR, "18_fuzzy_hybrid_recall.csv")
    df_hybrid_report.to_csv(hybrid_report_path, index=False)
    print(f"  --> Saved {hybrid_report_path}")

    # -------------------------------------------------------------
    # 4. Missed-GT Recovery Analysis
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"MISSED-GT RECOVERY ANALYSIS: {len(all_missed_gt):,} pairs ({len(all_missed_gt)/total_gt_all*100:.2f}%)")
    print("=" * 60)
    
    df_missed = pd.DataFrame(all_missed_gt)
    # Register in DuckDB to pull original text for analysis
    con.register('df_missed', df_missed)
    con.register('eval_s1_full', eval_s1_df)
    con.register('target_corpus_full', target_corpus_df)
    
    missed_samples = con.execute("""
        SELECT 
            m.s1_id, m.target_id, m.country,
            s.s1_name as s1_name, s.s1_addr as s1_addr,
            t.business_name as tgt_name, t.business_address as tgt_addr
        FROM df_missed m
        JOIN eval_s1_full s ON m.s1_id = s.s1_id
        JOIN target_corpus_full t ON m.target_id = t.entity_id
    """).fetch_df()
    
    # Analyze why missed:
    # 1. Missing target address?
    missed_samples['tgt_addr_missing'] = missed_samples['tgt_addr'].isna() | (missed_samples['tgt_addr'].str.lower().isin(['none', 'null', '']))
    # 2. Name length disparity
    missed_samples['s1_len'] = missed_samples['s1_name'].fillna('').str.len()
    missed_samples['tgt_len'] = missed_samples['tgt_name'].fillna('').str.len()
    missed_samples['extreme_len_diff'] = (missed_samples['s1_len'] / (missed_samples['tgt_len'] + 1) > 3) | (missed_samples['tgt_len'] / (missed_samples['s1_len'] + 1) > 3)
    
    pct_missing_addr = round(float(missed_samples['tgt_addr_missing'].mean() * 100), 2)
    pct_len_diff = round(float(missed_samples['extreme_len_diff'].mean() * 100), 2)
    
    print(f"  Total missed true pairs: {len(missed_samples):,} out of {total_gt_all:,}")
    print(f"  - Missed pairs with MISSING target address: {pct_missing_addr}%")
    print(f"  - Missed pairs with extreme name length disparity (acronyms/abbreviations): {pct_len_diff}%")
    
    print("\nSample of Missed Ground Truth Pairs (First 5):")
    for idx, r in missed_samples.head(5).iterrows():
        print(f"  [{r['country']}] S1: {r['s1_name']} || {r['s1_addr']}")
        print(f"       Tgt: {r['tgt_name']} || {r['tgt_addr']}\n")
        
    missed_out_path = os.path.join(REPORTS_DIR, "19_missed_gt_analysis.csv")
    missed_samples.to_csv(missed_out_path, index=False)
    print(f"  --> Saved {missed_out_path}")

    # -------------------------------------------------------------
    # 5. Candidate Pareto Curve (Varying K)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CANDIDATE PARETO CURVE (Varying Top-K per S1)")
    print("=" * 60)
    
    # Using the merged candidates, sort by score and test K cutoffs
    pareto_points = []
    
    # Convert all retrieved pairs to a scored dataframe
    scored_pairs = []
    for cntry, data in channels.items():
        q_ids = data['q_ids']
        t_ids = data['t_ids']
        
        # Merge scores per pair
        pair_scores = defaultdict(float)
        # Give exact matches top score
        for s1, tid in data['exact_name']:
            pair_scores[(s1, tid)] = max(pair_scores[(s1, tid)], 1.0)
        for s1, tid in data['exact_addr']:
            pair_scores[(s1, tid)] = max(pair_scores[(s1, tid)], 1.0)
            
        for q_i, t_i, s in data['name_ranked_results']:
            s1, tid = q_ids[q_i], t_ids[t_i]
            pair_scores[(s1, tid)] = max(pair_scores[(s1, tid)], s)
            
        for q_i, t_i, s in data['addr_ranked_results']:
            s1, tid = q_ids[q_i], t_ids[t_i]
            pair_scores[(s1, tid)] = max(pair_scores[(s1, tid)], s)
            
        for q_i, t_i, s in data['word_ranked_results']:
            s1, tid = q_ids[q_i], t_ids[t_i]
            pair_scores[(s1, tid)] = max(pair_scores[(s1, tid)], s)
            
        for (s1, tid), score in pair_scores.items():
            scored_pairs.append({'s1_id': s1, 'target_id': tid, 'score': score})
            
    df_scored = pd.DataFrame(scored_pairs)
    print(f"  Total unique scored pairs across all channels: {len(df_scored):,}")
    
    # Sort descending by score for each S1 and compute rank once
    df_scored = df_scored.sort_values(by=['s1_id', 'score'], ascending=[True, False]).reset_index(drop=True)
    df_scored['rank'] = df_scored.groupby('s1_id').cumcount() + 1
    
    for k_val in [1, 2, 5, 10, 15, 20, 30, 40, 50, 75]:
        top_k = df_scored[df_scored['rank'] <= k_val]
        top_k_set = set(zip(top_k['s1_id'], top_k['target_id']))
        k_hits = len(top_k_set.intersection(gt_pair_set))
        k_rec = round(float(k_hits / total_gt_all), 4)
        k_tot = len(top_k)
        avg_k = round(float(k_tot / total_q_all), 2)
        s1_dist = top_k.groupby('s1_id').size()
        p95_k = float(s1_dist.quantile(0.95))
        p99_k = float(s1_dist.quantile(0.99))
        
        pareto_points.append({
            'K_budget': k_val,
            'candidate_recall': k_rec,
            'hits': k_hits,
            'avg_candidates_per_s1': avg_k,
            'p95_candidates': p95_k,
            'p99_candidates': p99_k,
            'total_candidates': k_tot
        })
        print(f"  K = {k_val:>2d} | Candidate Recall: {k_rec*100:>6.2f}% | Avg Cand/S1: {avg_k:>5.2f} | P95: {p95_k:>4.1f} | Total: {k_tot:>7,d}")
        
    df_pareto_curve = pd.DataFrame(pareto_points)
    pareto_curve_path = os.path.join(REPORTS_DIR, "20_fuzzy_pareto_curve.csv")
    df_pareto_curve.to_csv(pareto_curve_path, index=False)
    print(f"  --> Saved {pareto_curve_path}")
    
    # Save candidates up to K=50 for XGBoost end-to-end evaluation
    print("\nSaving optimal top-50 candidate pairs for ML model training...")
    top_50_df = df_scored[df_scored['rank'] <= 50].copy()
    top_50_df['is_match'] = top_50_df.apply(lambda r: 1 if (r['s1_id'], r['target_id']) in gt_pair_set else 0, axis=1)
    cand_parquet_path = os.path.join(REPORTS_DIR, "21_improved_candidates.parquet")
    top_50_df.to_parquet(cand_parquet_path, index=False)
    print(f"  --> Saved {cand_parquet_path} ({len(top_50_df):,} pairs: {top_50_df['is_match'].sum():,} positives, {len(top_50_df)-top_50_df['is_match'].sum():,} negatives)")
    
    # Save exact ground truth dictionary and all S1 IDs in cohort
    gt_export = {}
    for _, r in eval_s1_df.iterrows():
        m = r['matched_entity_ids']
        gt_export[r['s1_id']] = [x for x in m.split(',') if x] if (pd.notna(m) and str(m).strip()) else []
    gt_dict_path = os.path.join(REPORTS_DIR, "21_eval_gt_dict.json")
    with open(gt_dict_path, "w", encoding="utf-8") as f:
        json.dump(gt_export, f)
    print(f"  --> Saved {gt_dict_path} ({len(gt_export):,} evaluation S1 entities)")
    
    print("\nSteps 1 - 4 Complete!")

if __name__ == "__main__":
    run_step07()
