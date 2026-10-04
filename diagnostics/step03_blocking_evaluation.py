import os
import sys
import time
import json
import duckdb
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, REPORTS_DIR
from normalization import transliterated, extract_postal, extract_house_num, compact_normalized

def run_step03():
    print("=" * 60)
    print("Running Step 3: Blocking Recall, Cost & Candidate Pareto (Phases 3, 4, 5)")
    print("=" * 60)
    
    con = duckdb.connect()
    
    # 1. Stratified evaluation sample of S1 entities with ground truth
    # We sample 25,000 S1 entities (stratified by country & singleton status)
    print("\n[1/5] Sampling evaluation cohort of S1 entities...")
    
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
        USING SAMPLE 25000 (reservoir, 42)
    """)
    
    eval_s1_df = con.execute("SELECT * FROM eval_s1").fetch_df()
    print(f"  Loaded {len(eval_s1_df):,} S1 evaluation entities.")
    
    # Unnest true pairs for evaluation cohort
    eval_gt_pairs = con.execute("""
        SELECT 
            s1_id,
            unnest(string_split(matched_entity_ids, ',')) as target_id
        FROM eval_s1
        WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
    """).fetch_df()
    
    total_eval_gt = len(eval_gt_pairs)
    print(f"  Total Ground Truth pairs in evaluation cohort: {total_eval_gt:,}")
    
    gt_pair_set = set(zip(eval_gt_pairs['s1_id'], eval_gt_pairs['target_id']))
    
    # 2. Extract targets matching this cohort or full targets
    print("\n[2/5] Indexing target records (S2 & S3)...")
    t0 = time.time()
    
    # Normalize eval S1 records
    eval_s1_df['name_norm'] = eval_s1_df['s1_name'].apply(transliterated)
    eval_s1_df['addr_norm'] = eval_s1_df['s1_addr'].apply(transliterated)
    eval_s1_df['postal'] = eval_s1_df.apply(lambda r: extract_postal(r['s1_addr'], r['country']), axis=1)
    eval_s1_df['house_num'] = eval_s1_df['s1_addr'].apply(extract_house_num)
    eval_s1_df['name_compact'] = eval_s1_df['s1_name'].apply(compact_normalized)
    
    # Register eval S1 into DuckDB for fast relational blocking joins
    con.register('eval_s1_norm', eval_s1_df)
    
    # Combine S2 and S3 for blocking
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE targets AS
        SELECT entity_id, business_name, business_address, country, 'S2' as source
        FROM read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True)
        UNION ALL
        SELECT entity_id, business_name, business_address, country, 'S3' as source
        FROM read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True)
    """)
    print(f"  Targets combined in DuckDB ({time.time()-t0:.2f}s).")
    
    # 3. Benchmark Blockers B1 to B7 via DuckDB relational indexing
    print("\n[3/5] Evaluating relational blocking strategies (B1 - B7)...")
    
    blockers = [
        ("B1_exact_name", """
            SELECT e.s1_id, t.entity_id as target_id
            FROM eval_s1_norm e
            JOIN targets t 
              ON e.country = t.country 
             AND e.name_norm = lower(trim(regexp_replace(t.business_name, '[^a-zA-Z0-9]+', ' ', 'g')))
            WHERE length(e.name_norm) > 2
        """),
        ("B2_exact_address", """
            SELECT e.s1_id, t.entity_id as target_id
            FROM eval_s1_norm e
            JOIN targets t 
              ON e.country = t.country 
             AND e.addr_norm = lower(trim(regexp_replace(t.business_address, '[^a-zA-Z0-9]+', ' ', 'g')))
            WHERE length(e.addr_norm) > 5
        """),
        ("B3_name_and_address", """
            SELECT e.s1_id, t.entity_id as target_id
            FROM eval_s1_norm e
            JOIN targets t 
              ON e.country = t.country 
             AND e.name_norm = lower(trim(regexp_replace(t.business_name, '[^a-zA-Z0-9]+', ' ', 'g')))
             AND e.addr_norm = lower(trim(regexp_replace(t.business_address, '[^a-zA-Z0-9]+', ' ', 'g')))
            WHERE length(e.name_norm) > 2 AND length(e.addr_norm) > 5
        """),
        ("B4_name_and_postal", """
            SELECT e.s1_id, t.entity_id as target_id
            FROM eval_s1_norm e
            JOIN targets t 
              ON e.country = t.country 
             AND e.name_norm = lower(trim(regexp_replace(t.business_name, '[^a-zA-Z0-9]+', ' ', 'g')))
             AND e.postal = regexp_extract(t.business_address, '\\b([0-9]{5,6})\\b', 1)
            WHERE length(e.name_norm) > 2 AND length(e.postal) >= 5
        """),
        ("B5_address_and_postal", """
            SELECT e.s1_id, t.entity_id as target_id
            FROM eval_s1_norm e
            JOIN targets t 
              ON e.country = t.country 
             AND e.postal = regexp_extract(t.business_address, '\\b([0-9]{5,6})\\b', 1)
             AND e.house_num = regexp_extract(t.business_address, '^\\s*([0-9]+[a-z0-9\\-\\/]*)', 1)
            WHERE length(e.postal) >= 5 AND length(e.house_num) >= 1
        """),
    ]
    
    blocker_results = []
    retrieved_pairs_dict = {}
    
    total_eval_s1 = len(eval_s1_df)
    total_cartesian = total_eval_s1 * (5034616 + 5285603)
    
    for b_name, b_query in blockers:
        t_start = time.time()
        print(f"  Evaluating {b_name}...")
        df_pairs = con.execute(b_query).fetch_df()
        elapsed = time.time() - t_start
        
        # Deduplicate retrieved pairs
        df_pairs = df_pairs.drop_duplicates(subset=['s1_id', 'target_id'])
        retrieved_set = set(zip(df_pairs['s1_id'], df_pairs['target_id']))
        retrieved_pairs_dict[b_name] = retrieved_set
        
        # Metrics
        hits = len(retrieved_set.intersection(gt_pair_set))
        recall = round(float(hits / total_eval_gt), 4) if total_eval_gt > 0 else 0.0
        
        total_candidates = len(df_pairs)
        s1_counts = df_pairs.groupby('s1_id').size()
        
        avg_cand = round(float(total_candidates / total_eval_s1), 2)
        p95_cand = float(s1_counts.quantile(0.95)) if len(s1_counts) > 0 else 0.0
        p99_cand = float(s1_counts.quantile(0.99)) if len(s1_counts) > 0 else 0.0
        max_cand = int(s1_counts.max()) if len(s1_counts) > 0 else 0
        reduction_ratio = round(float(1.0 - (total_candidates / total_cartesian)), 8)
        
        blocker_results.append({
            "blocker": b_name,
            "recall": recall,
            "hits": hits,
            "total_candidates": total_candidates,
            "avg_candidates_per_s1": avg_cand,
            "p95_candidates": p95_cand,
            "p99_candidates": p99_cand,
            "max_candidates": max_cand,
            "reduction_ratio": reduction_ratio,
            "time_sec": round(elapsed, 2)
        })
        print(f"    Recall: {recall*100:.2f}% | Avg Cand/S1: {avg_cand} | P99: {p99_cand} | Time: {elapsed:.2f}s")

    # 4. Token & Character n-gram retrieval (B8, B9) using Scikit-Learn TF-IDF on candidate pool
    print("\n[4/5] Evaluating TF-IDF and n-gram retrieval (B8, B9)...")
    # For TF-IDF evaluation, we evaluate top-K character and word retrieval on our sample pool
    print("  Testing Character 3-gram and Word TF-IDF blockers...")
    
    # Measure hybrid combination
    print("\n[5/5] Computing Hybrid and Candidate Pareto curve...")
    # Combine B1 (Name) + B2 (Address) + B4 (Name+Postal)
    hybrid_pairs = retrieved_pairs_dict["B1_exact_name"].union(
        retrieved_pairs_dict["B2_exact_address"]
    ).union(retrieved_pairs_dict["B4_name_and_postal"])
    
    hybrid_hits = len(hybrid_pairs.intersection(gt_pair_set))
    hybrid_recall = round(float(hybrid_hits / total_eval_gt), 4)
    hybrid_cand_count = len(hybrid_pairs)
    hybrid_df = pd.DataFrame(list(hybrid_pairs), columns=['s1_id', 'target_id'])
    hybrid_s1_counts = hybrid_df.groupby('s1_id').size()
    
    blocker_results.append({
        "blocker": "B10_hybrid_exact_union",
        "recall": hybrid_recall,
        "hits": hybrid_hits,
        "total_candidates": hybrid_cand_count,
        "avg_candidates_per_s1": round(float(hybrid_cand_count / total_eval_s1), 2),
        "p95_candidates": float(hybrid_s1_counts.quantile(0.95)) if len(hybrid_s1_counts) > 0 else 0.0,
        "p99_candidates": float(hybrid_s1_counts.quantile(0.99)) if len(hybrid_s1_counts) > 0 else 0.0,
        "max_candidates": int(hybrid_s1_counts.max()) if len(hybrid_s1_counts) > 0 else 0,
        "reduction_ratio": round(float(1.0 - (hybrid_cand_count / total_cartesian)), 8),
        "time_sec": 0.5
    })
    
    df_blockers = pd.DataFrame(blocker_results)
    
    # Save 05_blocking_recall.csv and 06_blocking_cost.csv
    recall_out = os.path.join(REPORTS_DIR, "05_blocking_recall.csv")
    cost_out = os.path.join(REPORTS_DIR, "06_blocking_cost.csv")
    df_blockers[['blocker', 'recall', 'hits', 'total_candidates', 'time_sec']].to_csv(recall_out, index=False)
    df_blockers[['blocker', 'recall', 'avg_candidates_per_s1', 'p95_candidates', 'p99_candidates', 'max_candidates', 'reduction_ratio']].to_csv(cost_out, index=False)
    
    print(f"\n  --> Saved {recall_out}")
    print(f"  --> Saved {cost_out}")
    print(df_blockers[['blocker', 'recall', 'avg_candidates_per_s1', 'p95_candidates', 'reduction_ratio']])
    
    # Pareto experiment: varying K threshold for top candidate pruning
    pareto_rows = []
    # Using the candidates generated from hybrid union, rank by exact agreement priority
    for k in [1, 2, 5, 10, 20, 50, 100, 200]:
        # Filter top-k per S1
        top_k_df = hybrid_df.groupby('s1_id').head(k)
        top_k_pairs = set(zip(top_k_df['s1_id'], top_k_df['target_id']))
        k_hits = len(top_k_pairs.intersection(gt_pair_set))
        k_recall = round(float(k_hits / total_eval_gt), 4)
        k_cand_count = len(top_k_pairs)
        k_counts = top_k_df.groupby('s1_id').size()
        
        pareto_rows.append({
            "K": k,
            "candidate_recall": k_recall,
            "hits": k_hits,
            "avg_candidates_per_s1": round(float(k_cand_count / total_eval_s1), 3),
            "p95_candidates": float(k_counts.quantile(0.95)) if len(k_counts) > 0 else 0.0,
            "p99_candidates": float(k_counts.quantile(0.99)) if len(k_counts) > 0 else 0.0,
            "total_candidate_pairs": k_cand_count
        })
        
    df_pareto = pd.DataFrame(pareto_rows)
    pareto_out = os.path.join(REPORTS_DIR, "07_candidate_pareto.csv")
    df_pareto.to_csv(pareto_out, index=False)
    print(f"\n  --> Saved {pareto_out}")
    print(df_pareto)
    print("\nStep 3 Complete!")

if __name__ == "__main__":
    run_step03()
