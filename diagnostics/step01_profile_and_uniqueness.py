import os
import sys
import json
import time
import duckdb
import pandas as pd
import numpy as np

# Ensure UTF-8 output and local imports
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TEST_S1, TEST_S2, TEST_S3, TRAIN_GT, REPORTS_DIR

def run_step01():
    print("=" * 60)
    print("Running Step 1: Data Profile, Uniqueness & Frequency Distributions")
    print("=" * 60)
    
    con = duckdb.connect()
    
    sources = {
        'S1_train': TRAIN_S1,
        'S2_train': TRAIN_S2,
        'S3_train': TRAIN_S3,
        'S1_test': TEST_S1,
        'S2_test': TEST_S2,
        'S3_test': TEST_S3,
    }
    
    # -------------------------------------------------------------
    # 1. Dataset Profile
    # -------------------------------------------------------------
    profile = {}
    print("\n[1/4] Profiling datasets...")
    for s_name, s_path in sources.items():
        t0 = time.time()
        res = con.execute(f"""
            SELECT 
                count(*) as total_rows,
                count(CASE WHEN business_address IS NULL OR lower(trim(business_address)) IN ('none', 'null', 'nan', '') THEN 1 END) as missing_address_count,
                count(CASE WHEN business_name IS NULL OR lower(trim(business_name)) IN ('none', 'null', 'nan', '') THEN 1 END) as missing_name_count
            FROM read_csv('{s_path}', delim='\\t', header=True, quote='', all_varchar=True)
        """).fetchone()
        
        # Country breakdown
        countries = con.execute(f"""
            SELECT country, count(*) as cnt
            FROM read_csv('{s_path}', delim='\\t', header=True, quote='', all_varchar=True)
            GROUP BY country
        """).fetchall()
        country_dict = {c: cnt for c, cnt in countries}
        
        profile[s_name] = {
            "total_rows": int(res[0]),
            "missing_address_count": int(res[1]),
            "missing_address_pct": round(float(res[1] / res[0] * 100), 3),
            "missing_name_count": int(res[2]),
            "country_distribution": country_dict,
            "profile_time_sec": round(time.time() - t0, 2)
        }
        print(f"  {s_name:10s}: {res[0]:>10,d} rows | Missing Addr: {res[1]:>8,d} ({res[1]/res[0]*100:.2f}%) | Countries: {country_dict}")

    # Ground truth profile
    gt_stats = con.execute(f"""
        SELECT 
            count(*) as total_s1,
            count(CASE WHEN matched_entity_ids IS NULL OR length(trim(matched_entity_ids)) = 0 THEN 1 END) as singletons,
            count(CASE WHEN matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0 THEN 1 END) as with_matches
        FROM read_csv('{TRAIN_GT}', delim='\\t', header=True, quote='', all_varchar=True)
    """).fetchone()
    
    # Unnest GT to count total true pairs
    gt_pair_count = con.execute(f"""
        SELECT count(*) FROM (
            SELECT unnest(string_split(matched_entity_ids, ','))
            FROM read_csv('{TRAIN_GT}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
        )
    """).fetchone()[0]
    
    profile["GroundTruth_train"] = {
        "total_s1": int(gt_stats[0]),
        "singletons": int(gt_stats[1]),
        "singleton_pct": round(float(gt_stats[1] / gt_stats[0] * 100), 2),
        "with_matches": int(gt_stats[2]),
        "with_matches_pct": round(float(gt_stats[2] / gt_stats[0] * 100), 2),
        "total_true_pairs": int(gt_pair_count),
        "avg_matches_per_matched_s1": round(float(gt_pair_count / gt_stats[2]), 3)
    }
    
    profile_out = os.path.join(REPORTS_DIR, "01_dataset_profile.json")
    with open(profile_out, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"  --> Saved {profile_out}")

    # -------------------------------------------------------------
    # 2. Uniqueness Tests (Phase 1)
    # -------------------------------------------------------------
    print("\n[2/4] Computing uniqueness metrics for S1, S2, S3 (Train)...")
    uniqueness_rows = []
    train_sources = [('S1', TRAIN_S1), ('S2', TRAIN_S2), ('S3', TRAIN_S3)]
    
    for s_label, s_path in train_sources:
        t0 = time.time()
        print(f"  Processing {s_label}...")
        
        # Regex clean rules:
        # norm_name: lowercase, alpha-num with spaces
        # norm_addr: lowercase, alpha-num with spaces
        # postal: 5 or 6 digits
        q = f"""
            SELECT 
                count(*) as total_rows,
                count(distinct business_name) as raw_name_uniq,
                count(distinct lower(trim(regexp_replace(business_name, '[^a-zA-Z0-9]+', ' ', 'g')))) as norm_name_uniq,
                count(distinct business_address) as raw_addr_uniq,
                count(distinct lower(trim(regexp_replace(business_address, '[^a-zA-Z0-9]+', ' ', 'g')))) as norm_addr_uniq,
                count(distinct concat(
                    lower(trim(regexp_replace(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), 
                    '||', 
                    lower(trim(regexp_replace(business_address, '[^a-zA-Z0-9]+', ' ', 'g')))
                )) as name_addr_uniq,
                count(distinct concat(
                    lower(trim(regexp_replace(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), 
                    '||', 
                    COALESCE(regexp_extract(business_address, '\\b([0-9]{{5,6}})\\b', 1), '')
                )) as name_postal_uniq,
                count(distinct concat(
                    lower(trim(regexp_replace(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), 
                    '||', 
                    COALESCE(regexp_extract(business_address, '\\b([0-9]{{5,6}})\\b', 1), '')
                )) as addr_postal_uniq
            FROM read_csv('{s_path}', delim='\\t', header=True, quote='', all_varchar=True)
        """
        row = con.execute(q).fetchone()
        tot = row[0]
        
        metrics = [
            ("Raw name uniqueness", row[1]),
            ("Normalized name uniqueness", row[2]),
            ("Raw address uniqueness", row[3]),
            ("Normalized address uniqueness", row[4]),
            ("Name+address uniqueness", row[5]),
            ("Name+postal uniqueness", row[6]),
            ("Address+postal uniqueness", row[7]),
        ]
        
        for m_name, m_val in metrics:
            uniqueness_rows.append({
                "source": s_label,
                "test": m_name,
                "unique_values": int(m_val),
                "total_rows": int(tot),
                "uniqueness_ratio": round(float(m_val / tot), 6),
                "uniqueness_pct": round(float(m_val / tot * 100), 2)
            })
        print(f"  {s_label} computed in {time.time()-t0:.2f}s")
        
    df_uniqueness = pd.DataFrame(uniqueness_rows)
    uniqueness_out = os.path.join(REPORTS_DIR, "02_uniqueness.csv")
    df_uniqueness.to_csv(uniqueness_out, index=False)
    print(f"  --> Saved {uniqueness_out}")
    print(df_uniqueness.pivot(index='test', columns='source', values='uniqueness_pct'))

    # -------------------------------------------------------------
    # 3. Frequency Distributions (Phase 1)
    # -------------------------------------------------------------
    print("\n[3/4] Computing frequency distributions and quantiles...")
    freq_rows = []
    
    for s_label, s_path in train_sources:
        print(f"  Analyzing distributions for {s_label}...")
        fields_to_check = [
            ('normalized_name', "lower(trim(regexp_replace(business_name, '[^a-zA-Z0-9]+', ' ', 'g')))"),
            ('normalized_address', "lower(trim(regexp_replace(business_address, '[^a-zA-Z0-9]+', ' ', 'g')))"),
            ('name_plus_address', "concat(lower(trim(regexp_replace(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), '||', lower(trim(regexp_replace(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))))"),
        ]
        for field, col_expr in fields_to_check:
            q_freq = f"""
                WITH counts AS (
                    SELECT {col_expr} as val, count(*) as freq
                    FROM read_csv('{s_path}', delim='\\t', header=True, quote='', all_varchar=True)
                    WHERE {col_expr} IS NOT NULL AND length({col_expr}) > 0
                    GROUP BY 1
                )
                SELECT 
                    avg(freq) as avg_freq,
                    median(freq) as p50_freq,
                    quantile_cont(freq, 0.75) as p75_freq,
                    quantile_cont(freq, 0.90) as p90_freq,
                    quantile_cont(freq, 0.95) as p95_freq,
                    quantile_cont(freq, 0.99) as p99_freq,
                    max(freq) as max_freq,
                    count(CASE WHEN freq = 1 THEN 1 END) * 1.0 / count(*) as singleton_rate
                FROM counts
            """
            q_res = con.execute(q_freq).fetchone()
            freq_rows.append({
                "source": s_label,
                "field": field,
                "avg_freq": round(float(q_res[0]), 3),
                "p50_freq": float(q_res[1]),
                "p75_freq": float(q_res[2]),
                "p90_freq": float(q_res[3]),
                "p95_freq": float(q_res[4]),
                "p99_freq": float(q_res[5]),
                "max_freq": int(q_res[6]),
                "val_singleton_rate": round(float(q_res[7]), 4)
            })

    df_freq = pd.DataFrame(freq_rows)
    freq_out = os.path.join(REPORTS_DIR, "03_frequency_distributions.csv")
    df_freq.to_csv(freq_out, index=False)
    print(f"  --> Saved {freq_out}")
    print(df_freq[['source', 'field', 'avg_freq', 'p50_freq', 'p90_freq', 'p99_freq', 'max_freq']])

    # -------------------------------------------------------------
    # 4. Token Frequency Analysis (Names & Addresses)
    # -------------------------------------------------------------
    print("\n[4/4] Inspecting top and rare tokens on S1...")
    top_name_tokens = con.execute(f"""
        WITH tokens AS (
            SELECT unnest(string_split(lower(trim(regexp_replace(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) as tok
            FROM read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True)
        )
        SELECT tok, count(*) as df
        FROM tokens
        WHERE length(tok) >= 2
        GROUP BY tok
        ORDER BY df DESC
        LIMIT 20
    """).df()
    print("Top 10 business name tokens in S1:")
    print(top_name_tokens.head(10))

    top_addr_tokens = con.execute(f"""
        WITH tokens AS (
            SELECT unnest(string_split(lower(trim(regexp_replace(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) as tok
            FROM read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE business_address IS NOT NULL
        )
        SELECT tok, count(*) as df
        FROM tokens
        WHERE length(tok) >= 2
        GROUP BY tok
        ORDER BY df DESC
        LIMIT 20
    """).df()
    print("\nTop 10 address tokens in S1:")
    print(top_addr_tokens.head(10))

    print("\nStep 1 Complete! Artifacts generated.")

if __name__ == "__main__":
    run_step01()
