"""
Business Entity Resolution — End-to-End Scalable Production Pipeline
=====================================================================
Executes the full Entity Resolution workflow:
1. Ingestion of Source 1, Source 2, and Source 3 TSVs via DuckDB
2. Deterministic multi-view normalization (transliteration, state expansion, legal suffixes)
3. Country-partitioned inverted index retrieval (Word TF-IDF + exact fallback)
4. RapidFuzz SIMD pairwise feature computation
5. XGBoost scoring with optimal thresholding (0.70) and singleton pruning
6. Streaming export of candidate_pairs.tsv and matching_results.tsv

Ensures 100% compliance with official submission validator requirements.
"""

import os
import sys
import time
import argparse
import duckdb
import numpy as np
import pandas as pd
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalize import build_normalized_record
from retrieval import CountryIndex
from features import compute_batch_features
from model import (
    train_model, save_model, load_model,
    predict_pair_probabilities, apply_decision_policy,
    evaluate_macro_f05, DECISION_THRESHOLD, SINGLETON_FLOOR
)


def load_and_normalize_records(tsv_path: str) -> list:
    """Load records from TSV and normalize using DuckDB and python multi-view normalization."""
    print(f"  Reading {os.path.basename(tsv_path)}...")
    t0 = time.time()
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{tsv_path}', delim='\\t', header=True, quote='', all_varchar=True)
    """).fetch_df()
    
    print(f"    Loaded {len(df):,} records ({time.time()-t0:.2f}s). Normalizing text...")
    t0 = time.time()
    
    records = []
    # Fast row iteration
    for row in df.itertuples(index=False):
        records.append(build_normalized_record(
            raw_id=row.entity_id,
            raw_name=row.business_name,
            raw_addr=row.business_address,
            country=row.country
        ))
        
    print(f"    Normalized {len(records):,} records in {time.time()-t0:.2f}s.")
    return records


def prepare_training_pairs(train_dir: str, sample_size: int = 150000) -> tuple:
    """
    Extract high-quality true matches and hard negative pairs from training data
    to train the XGBoost candidate ranker.
    """
    print("\n[TRAIN PREP] Sampling balanced training pairs from Ground Truth...")
    con = duckdb.connect()
    
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    
    t0 = time.time()
    # 1. Sample true positive pairs
    query = f"""
        WITH sample_gt AS (
            SELECT 
                source1_entity_id as s1_id,
                unnest(string_split(matched_entity_ids, ',')) as target_id
            FROM read_csv('{gt_path}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
            USING SAMPLE {sample_size // 2} (reservoir, 42)
        ),
        all_targets AS (
            SELECT entity_id, business_name, business_address, country FROM read_csv('{s2_path}', delim='\\t', header=True, quote='', all_varchar=True)
            UNION ALL
            SELECT entity_id, business_name, business_address, country FROM read_csv('{s3_path}', delim='\\t', header=True, quote='', all_varchar=True)
        )
        SELECT 
            g.s1_id,
            s1.business_name as s1_name,
            s1.business_address as s1_addr,
            s1.country as s1_country,
            g.target_id,
            t.business_name as tgt_name,
            t.business_address as tgt_addr,
            1.0 as label
        FROM sample_gt g
        JOIN read_csv('{s1_path}', delim='\\t', header=True, quote='', all_varchar=True) s1 ON g.s1_id = s1.entity_id
        JOIN all_targets t ON g.target_id = t.entity_id
    """
    pos_df = con.execute(query).fetch_df()
    print(f"  Sampled {len(pos_df):,} true positive pairs ({time.time()-t0:.2f}s).")
    
    # 2. Sample hard negative pairs (name or address overlap but different entity)
    t0 = time.time()
    neg_query = f"""
        WITH sampled_s1 AS (
            SELECT entity_id as s1_id, business_name as s1_name, business_address as s1_addr, country as s1_country
            FROM read_csv('{s1_path}', delim='\\t', header=True, quote='', all_varchar=True)
            USING SAMPLE 20000 (reservoir, 99)
        ),
        all_targets AS (
            SELECT entity_id as target_id, business_name as tgt_name, business_address as tgt_addr, country as tgt_country 
            FROM read_csv('{s2_path}', delim='\\t', header=True, quote='', all_varchar=True)
            USING SAMPLE 200000 (reservoir, 99)
        ),
        gt_pairs AS (
            SELECT source1_entity_id as s1_id, unnest(string_split(matched_entity_ids, ',')) as target_id
            FROM read_csv('{gt_path}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
        )
        SELECT 
            s.s1_id, s.s1_name, s.s1_addr, s.s1_country,
            t.target_id, t.tgt_name, t.tgt_addr,
            0.0 as label
        FROM sampled_s1 s
        JOIN all_targets t ON s.s1_country = t.tgt_country
          AND (substring(s.s1_name, 1, 5) = substring(t.tgt_name, 1, 5) OR substring(s.s1_addr, 1, 6) = substring(t.tgt_addr, 1, 6))
        LEFT JOIN gt_pairs g ON s.s1_id = g.s1_id AND t.target_id = g.target_id
        WHERE g.target_id IS NULL
        USING SAMPLE {len(pos_df)} (reservoir, 42)
    """
    neg_df = con.execute(neg_query).fetch_df()
    print(f"  Sampled {len(neg_df):,} hard negative pairs ({time.time()-t0:.2f}s).")
    
    train_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Compute features for all pairs
    features_input = []
    from normalize import normalize_name, normalize_address, extract_postal, extract_house_num
    
    for row in train_df.itertuples(index=False):
        n1 = normalize_name(row.s1_name)
        n2 = normalize_name(row.tgt_name)
        a1 = normalize_address(row.s1_addr, row.s1_country)
        a2 = normalize_address(row.tgt_addr, row.s1_country)
        p1 = extract_postal(row.s1_addr, row.s1_country)
        p2 = extract_postal(row.tgt_addr, row.s1_country)
        h1 = extract_house_num(row.s1_addr)
        h2 = extract_house_num(row.tgt_addr)
        # Approximate retrieval score based on token overlap
        ret_s = len(set(n1.split()) & set(n2.split())) / max(1, len(set(n1.split()) | set(n2.split())))
        features_input.append((n1, n2, a1, a2, p1, p2, h1, h2, ret_s))
        
    X_train = compute_batch_features(features_input)
    y_train = train_df['label'].values.astype(np.float32)
    
    return X_train, y_train


def run_pipeline(data_dir: str, output_dir: str, model_path: str, k: int = 30, mode: str = "test", train_dir: str = None, sample_limit: int = None):
    """Execute end-to-end entity resolution inference and output generation."""
    print("=" * 70)
    print(f"AMAZON ML CHALLENGE — PRODUCTION ENTITY RESOLUTION PIPELINE")
    print(f"Mode: {mode.upper()} | Candidate K: {k} | Output Dir: {output_dir}")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    
    # -----------------------------------------------------------------
    # Step 1: Model Management (Train or Load)
    # -----------------------------------------------------------------
    if os.path.exists(model_path) and mode not in ("retrain", "train"):
        print(f"\n[1/4] Loading trained XGBoost model from {model_path}...")
        model = load_model(model_path)
    else:
        print(f"\n[1/4] Training new XGBoost matcher from training dataset...")
        t_dir = train_dir or data_dir
        X_train, y_train = prepare_training_pairs(t_dir, sample_size=120000)
        t0 = time.time()
        model = train_model(X_train, y_train)
        print(f"  Model trained in {time.time()-t0:.2f}s. Saving to {model_path}...")
        save_model(model, model_path)
        if mode == "train":
            print(f"\nTraining completed. Model saved to {model_path}.")
            return
        
    # -----------------------------------------------------------------
    # Step 2 & 3: Country-Partitioned Ingestion, Inverted Indexing & Scoring
    # -----------------------------------------------------------------
    prefix = "test" if mode == "test" else "train"
    s1_file = os.path.join(data_dir, f"{prefix}_source1.tsv")
    s2_file = os.path.join(data_dir, f"{prefix}_source2.tsv")
    s3_file = os.path.join(data_dir, f"{prefix}_source3.tsv")
    
    if not os.path.exists(s1_file):
        raise FileNotFoundError(f"Source 1 file not found at {s1_file}")
        
    matching_tsv_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    con = duckdb.connect()
    country_rows = con.execute(f"""
        SELECT DISTINCT country 
        FROM read_csv('{s1_file}', delim='\\t', header=True, quote='', all_varchar=True)
        WHERE country IS NOT NULL AND length(trim(country)) > 0
    """).fetchall()
    countries = [r[0] for r in country_rows]
    print(f"\n[2/4] Discovered {len(countries)} country partition(s): {', '.join(countries)}")
    
    total_s1 = con.execute(f"SELECT count(*) FROM read_csv('{s1_file}', delim='\\t', header=True, quote='', all_varchar=True)").fetchone()[0]
    print(f"  Total S1 query records across all countries: {total_s1:,}")
    
    print(f"\n[3/4] Running country-partitioned retrieval & inference (K={k})...")
    
    processed_count = 0
    total_cands_written = 0
    total_matches_written = 0
    
    import gc
    
    with open(matching_tsv_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_tsv_path, 'w', encoding='utf-8') as f_cand:
        
        # Write official headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for country in countries:
            print(f"\n=======================================================")
            print(f"Processing Country Partition: {country}")
            print(f"=======================================================")
            
            # Load S1 queries for this country
            t0 = time.time()
            limit_clause = f"LIMIT {sample_limit}" if sample_limit else ""
            q_df = con.execute(f"""
                SELECT entity_id, business_name, business_address, country
                FROM read_csv('{s1_file}', delim='\\t', header=True, quote='', all_varchar=True)
                WHERE country = '{country}'
                {limit_clause}
            """).fetch_df()
            q_list = [build_normalized_record(r.entity_id, r.business_name, r.business_address, r.country)
                      for r in q_df.itertuples(index=False)]
            del q_df
            print(f"  Loaded & normalized {len(q_list):,} S1 queries ({time.time()-t0:.2f}s).")
            
            # Load S2 and S3 targets for this country
            t0 = time.time()
            t_df = con.execute(f"""
                SELECT entity_id, business_name, business_address, country
                FROM read_csv('{s2_file}', delim='\\t', header=True, quote='', all_varchar=True)
                WHERE country = '{country}'
                UNION ALL
                SELECT entity_id, business_name, business_address, country
                FROM read_csv('{s3_file}', delim='\\t', header=True, quote='', all_varchar=True)
                WHERE country = '{country}'
            """).fetch_df()
            t_list = [build_normalized_record(r.entity_id, r.business_name, r.business_address, r.country)
                      for r in t_df.itertuples(index=False)]
            del t_df
            print(f"  Loaded & normalized {len(t_list):,} S2/S3 targets ({time.time()-t0:.2f}s).")
            
            if not t_list:
                print(f"  Warning: No targets found for country {country}. Writing empty candidates.")
                for q in q_list:
                    f_cand.write(f"{q['entity_id']}\t\n")
                    f_match.write(f"{q['entity_id']}\t\n")
                processed_count += len(q_list)
                del q_list
                gc.collect()
                continue
                
            # Build target lookup map for this country only
            target_lookup = {r['entity_id']: r for r in t_list}
            
            # Build inverted index for this country
            t0 = time.time()
            c_index = CountryIndex(country, t_list)
            print(f"  Inverted index built in {time.time()-t0:.2f}s.")
            
            # Process queries in batches
            batch_size = 5000
            for b_start in range(0, len(q_list), batch_size):
                b_queries = q_list[b_start:b_start + batch_size]
                
                # Retrieve candidates
                cands_dict = c_index.retrieve_candidates(b_queries, k=k, batch_size=2000)
                
                # Prepare features for all candidates in this batch
                pair_meta = []
                pair_features_input = []
                
                for q in b_queries:
                    s1_id = q['entity_id']
                    cands = cands_dict.get(s1_id, [])
                    for tid, ret_score in cands:
                        tgt = target_lookup.get(tid)
                        if tgt:
                            pair_meta.append((s1_id, tid))
                            pair_features_input.append((
                                q['norm_name'], tgt['norm_name'],
                                q['norm_addr'], tgt['norm_addr'],
                                q['postal'], tgt['postal'],
                                q['house_num'], tgt['house_num'],
                                ret_score
                            ))
                            
                # Score candidates with XGBoost
                if pair_features_input:
                    X_batch = compute_batch_features(pair_features_input)
                    probs = predict_pair_probabilities(model, X_batch)
                else:
                    probs = np.array([])
                    
                # Group probabilities by s1_id
                s1_scored_cands = defaultdict(list)
                for (s1_id, tid), prob in zip(pair_meta, probs):
                    s1_scored_cands[s1_id].append((tid, float(prob)))
                    
                # Apply decision policy and write outputs for batch
                for q in b_queries:
                    s1_id = q['entity_id']
                    cands = cands_dict.get(s1_id, [])
                    cand_ids = [tid for tid, _ in cands]
                    
                    scored_cands = s1_scored_cands.get(s1_id, [])
                    matched_ids = apply_decision_policy(
                        scored_cands,
                        threshold=DECISION_THRESHOLD,
                        singleton_floor=SINGLETON_FLOOR
                    )
                    
                    # Ensure candidate_pairs.tsv contains comma-separated IDs
                    cand_str = ",".join(cand_ids)
                    f_cand.write(f"{s1_id}\t{cand_str}\n")
                    total_cands_written += len(cand_ids)
                    
                    # Ensure matching_results.tsv contains comma-separated IDs
                    match_str = ",".join(matched_ids)
                    f_match.write(f"{s1_id}\t{match_str}\n")
                    total_matches_written += len(matched_ids)
                    
                processed_count += len(b_queries)
                if processed_count % 25000 == 0 or processed_count == total_s1:
                    print(f"  Processed {processed_count:,}/{total_s1:,} queries "
                          f"({processed_count/total_s1*100:.1f}%)...")
                          
    print(f"\n[4/4] Pipeline Execution Completed Successfully!")
    print(f"  -> Matching results saved to: {matching_tsv_path}")
    print(f"  -> Candidate pairs saved to: {candidate_tsv_path}")
    print(f"  -> Total S1 queries written: {processed_count:,}")
    print(f"  -> Total candidate pairs: {total_cands_written:,} (Avg {total_cands_written/total_s1:.2f}/S1)")
    print(f"  -> Total final matches: {total_matches_written:,} (Avg {total_matches_written/total_s1:.2f}/S1)")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge — Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default="6ab10eb3b23ba_student_resource/student_resource/dataset/test",
                        help="Path to folder with source1/2/3 TSV files")
    parser.add_argument("--train-dir", default="6ab10eb3b23ba_student_resource/student_resource/dataset/train",
                        help="Path to training folder (for model fitting)")
    parser.add_argument("--output-dir", default="output",
                        help="Folder where matching_results.tsv and candidate_pairs.tsv are saved")
    parser.add_argument("--model-path", default="models/xgboost_matcher.joblib",
                        help="Path to save or load trained XGBoost model")
    parser.add_argument("--k", type=int, default=30,
                        help="Candidate budget per S1 record (30 for efficiency, 50 for high recall)")
    parser.add_argument("--mode", default="test", choices=["test", "train", "retrain"],
                        help="Pipeline run mode")
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Optional limit on queries per country for fast testing")
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        k=args.k,
        mode=args.mode,
        train_dir=args.train_dir,
        sample_limit=args.sample_limit
    )


if __name__ == "__main__":
    main()
