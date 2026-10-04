import os
import sys
import time
import duckdb
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, REPORTS_DIR
from normalization import raw, light_normalized, token_normalized, compact_normalized, transliterated

def run_step02():
    print("=" * 60)
    print("Running Step 2: Normalization Experiments (Phase 2)")
    print("=" * 60)
    
    con = duckdb.connect()
    
    # We sample a statistically robust subset of pairs (e.g. 50,000 true match pairs + S1/S2 records)
    # to evaluate collision rates, uniqueness changes, and GT recall across views.
    print("[1/3] Loading sample of true matching pairs for normalization evaluation...")
    
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sample_gt AS
        WITH unnested AS (
            SELECT 
                source1_entity_id, 
                unnest(string_split(matched_entity_ids, ',')) as target_id
            FROM read_csv('{TRAIN_GT}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
        )
        SELECT * FROM unnested
        USING SAMPLE 50000 (reservoir, 42)
    """)
    
    # Fetch paired records
    print("[2/3] Extracting paired text for S1 and Targets...")
    pairs_df = con.execute(f"""
        SELECT 
            g.source1_entity_id,
            g.target_id,
            s1.business_name as s1_name,
            s1.business_address as s1_addr,
            s1.country as s1_country,
            COALESCE(s2.business_name, s3.business_name) as target_name,
            COALESCE(s2.business_address, s3.business_address) as target_addr,
            COALESCE(s2.country, s3.country) as target_country
        FROM sample_gt g
        JOIN read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True) s1 ON g.source1_entity_id = s1.entity_id
        LEFT JOIN read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True) s2 ON g.target_id = s2.entity_id
        LEFT JOIN read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True) s3 ON g.target_id = s3.entity_id
    """).fetch_df()
    
    print(f"  Loaded {len(pairs_df):,} true pairs.")
    
    # Also sample 50,000 S1 records to measure uniqueness & collision rate
    s1_sample = con.execute(f"""
        SELECT business_name, business_address, country
        FROM read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True)
        USING SAMPLE 50000 (reservoir, 42)
    """).fetch_df()
    
    # Define normalization views
    views = {
        'raw': lambda s: raw(s),
        'light_normalized': lambda s: light_normalized(s),
        'token_normalized': lambda s: token_normalized(s),
        'compact_normalized': lambda s: compact_normalized(s),
        'transliterated': lambda s: transliterated(s),
    }
    
    results = []
    print("\n[3/3] Evaluating normalization views...")
    
    for v_name, norm_fn in views.items():
        t0 = time.time()
        print(f"  Testing view: {v_name}...")
        
        # 1. Uniqueness and collisions on S1 sample
        s1_names = s1_sample['business_name'].apply(norm_fn)
        s1_addrs = s1_sample['business_address'].apply(norm_fn)
        
        total_n = len(s1_sample)
        unique_names = s1_names.nunique()
        unique_addrs = s1_addrs.nunique()
        name_collision_rate = round(float((total_n - unique_names) / total_n), 4)
        addr_collision_rate = round(float((total_n - unique_addrs) / total_n), 4)
        
        # Frequency metrics
        name_counts = s1_names.value_counts()
        avg_name_freq = round(float(name_counts.mean()), 3)
        max_name_freq = int(name_counts.max())
        
        addr_counts = s1_addrs.value_counts()
        avg_addr_freq = round(float(addr_counts.mean()), 3)
        max_addr_freq = int(addr_counts.max())
        
        # 2. Ground truth recall on pairs
        s1_p_names = pairs_df['s1_name'].apply(norm_fn)
        tgt_p_names = pairs_df['target_name'].apply(norm_fn)
        name_exact_matches = (s1_p_names == tgt_p_names) & (s1_p_names != "")
        name_recall = round(float(name_exact_matches.sum() / len(pairs_df)), 4)
        
        s1_p_addrs = pairs_df['s1_addr'].apply(norm_fn)
        tgt_p_addrs = pairs_df['target_addr'].apply(norm_fn)
        addr_exact_matches = (s1_p_addrs == tgt_p_addrs) & (s1_p_addrs != "")
        addr_recall = round(float(addr_exact_matches.sum() / len(pairs_df)), 4)
        
        # Both match
        both_match = name_exact_matches & addr_exact_matches
        both_recall = round(float(both_match.sum() / len(pairs_df)), 4)
        
        # Either match
        either_match = name_exact_matches | addr_exact_matches
        either_recall = round(float(either_match.sum() / len(pairs_df)), 4)
        
        results.append({
            "view": v_name,
            "unique_names_sample": unique_names,
            "name_collision_rate": name_collision_rate,
            "avg_name_freq": avg_name_freq,
            "max_name_freq": max_name_freq,
            "name_exact_gt_recall": name_recall,
            "unique_addrs_sample": unique_addrs,
            "addr_collision_rate": addr_collision_rate,
            "avg_addr_freq": avg_addr_freq,
            "max_addr_freq": max_addr_freq,
            "addr_exact_gt_recall": addr_recall,
            "both_exact_gt_recall": both_recall,
            "either_exact_gt_recall": either_recall,
            "time_sec": round(time.time() - t0, 2)
        })
        
    df_results = pd.DataFrame(results)
    out_file = os.path.join(REPORTS_DIR, "04_normalization_comparison.csv")
    df_results.to_csv(out_file, index=False)
    print(f"\n  --> Saved {out_file}")
    print(df_results[['view', 'name_exact_gt_recall', 'addr_exact_gt_recall', 'either_exact_gt_recall', 'name_collision_rate', 'addr_collision_rate']])
    print("\nStep 2 Complete!")

if __name__ == "__main__":
    run_step02()
