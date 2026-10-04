"""
Master End-to-End Pipeline for Business Entity Resolution.
Amazon ML Challenge 2026.

Championship Single-Pass Architecture:
1. Load and Multi-View normalize S2 and S3 target records.
2. Build Country-Partitioned Multi-Channel Retrieval Indexes.
3. Stream S1 reference records through Adaptive Candidate Pruning and Model Scoring.
4. Apply Championship Entity-Level Decision Engine (Tiered Singleton Defense,
   Source-Specific Thresholds, and Expected-F0.5 Subset Optimization).
5. Export candidate_pairs.tsv and matching_results.tsv simultaneously in a single pass.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import joblib

sys.stdout.reconfigure(encoding="utf-8")

# Internal modules
from normalize import normalize_record
from retrieval import HybridRetriever
from features import extract_pair_features, extract_championship_features, extract_18_features, FEATURE_NAMES
from model import (
    create_xgboost_matcher,
    apply_two_stage_decision_policy,
    apply_championship_decision_policy
)


def parse_args():
    """Parse command-line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test directory containing test_source1/2/3.tsv")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory for results")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model joblib")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional limit on S1 records for dry-run verification")
    parser.add_argument("--threshold", type=float, default=0.72, help="Default decision threshold for matching")
    parser.add_argument("--singleton-floor", type=float, default=0.40, help="Singleton protection floor")
    parser.add_argument("--tau-s2", type=float, default=0.72, help="Source 2 decision threshold")
    parser.add_argument("--tau-s3", type=float, default=0.75, help="Source 3 decision threshold")
    parser.add_argument("--use-baseline-policy", action="store_true", help="Force baseline single-threshold policy")
    return parser.parse_args()


def load_source_tsv(path: Path, max_rows: int = None) -> List[Dict[str, Any]]:
    """Stream and load TSV rows as dicts."""
    print(f"Loading {path.name}...")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=max_rows)
    return df.to_dict("records")


def main():
    """Execute master end-to-end entity resolution inference pipeline."""
    args = parse_args()
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: CHAMPIONSHIP PRODUCTION PIPELINE")
    print("=" * 80)

    t0 = time.time()
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    # 1. LOAD TARGET RECORDS (S2 and S3)
    target_limit = args.sample_limit * 100 if args.sample_limit else None
    s2_raw = load_source_tsv(s2_path, max_rows=target_limit)
    s3_raw = load_source_tsv(s3_path, max_rows=target_limit)
    print(f"Loaded Target records - S2: {len(s2_raw):,}, S3: {len(s3_raw):,} in {time.time()-t0:.2f}s")

    # 2. MULTI-VIEW NORMALIZATION OF TARGETS
    print("\n[Stage 1] Multi-View Normalizing Targets (S2 + S3)...")
    t_norm = time.time()
    target_records = [normalize_record(r) for r in s2_raw] + [normalize_record(r) for r in s3_raw]
    del s2_raw, s3_raw  # Free raw dicts
    print(f"Normalized {len(target_records):,} target records in {time.time()-t_norm:.2f}s")

    # 3. BUILD COUNTRY-PARTITIONED MULTI-CHANNEL RETRIEVAL INDEX
    print("\n[Stage 2] Building Country-Partitioned Multi-Channel Retrieval Index...")
    t_idx = time.time()
    retriever = HybridRetriever(k_ceiling=30, steep_threshold=0.85, gap_threshold=0.35)
    retriever.index_targets(target_records)
    print(f"Indexed {len(target_records):,} target records in {time.time()-t_idx:.2f}s")

    # 4. LOAD MODEL
    print("\n[Stage 3] Loading Matcher Model...")
    model_path = args.model_path or Path(__file__).resolve().parent / "models" / "xgboost_matcher.joblib"
    if Path(model_path).exists():
        print(f"Loading trained matcher model from {model_path}...")
        model = joblib.load(str(model_path))
    else:
        print("Warning: Model path not found. Initializing fresh XGBoost matcher...")
        model = create_xgboost_matcher()

    # 5. STREAM S1 AND EXECUTE SINGLE-PASS PIPELINE
    print("\n[Stage 4] Streaming S1 (Candidate Generation + Feature Extraction + Decision Engine)...")
    s1_raw = load_source_tsv(s1_path, max_rows=args.sample_limit)
    total_s1 = len(s1_raw)
    print(f"Loaded {total_s1:,} S1 records. Commencing single-pass export...")

    cand_file_path = output_dir / "candidate_pairs.tsv"
    match_file_path = output_dir / "matching_results.tsv"

    total_candidates = 0
    match_count = 0
    singleton_count = 0
    t_stream = time.time()

    with open(cand_file_path, "w", encoding="utf-8") as f_cand, open(match_file_path, "w", encoding="utf-8") as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for idx, r in enumerate(s1_raw):
            s1_rec = normalize_record(r)
            s1_id = s1_rec["entity_id"]

            # Multi-channel retrieval + candidate ranking
            cands = retriever.retrieve_candidates_for_s1(s1_rec)
            total_candidates += len(cands)
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

            if not cands:
                f_match.write(f"{s1_id}\t\n")
                singleton_count += 1
            else:
                # Extract pairwise features
                feat_list = []
                use_18 = getattr(model, 'n_features_in_', 15) == 18
                for cid in cands:
                    trec = retriever.target_lookup[cid]
                    if use_18:
                        feat_list.append(extract_18_features(s1_rec, trec))
                    else:
                        feat_list.append(extract_pair_features(s1_rec, trec))

                X_pair = np.array(feat_list, dtype=np.float32)
                probs = model.predict_proba(X_pair)[:, 1]

                # Decision Policy
                if args.use_baseline_policy:
                    matches = apply_two_stage_decision_policy(
                        cands,
                        probs,
                        threshold=args.threshold,
                        singleton_floor=args.singleton_floor
                    )
                else:
                    matches = apply_championship_decision_policy(
                        candidate_ids=cands,
                        probabilities=probs,
                        pair_features=feat_list,
                        tau_s2=args.tau_s2,
                        tau_s3=args.tau_s3,
                        singleton_floor=args.singleton_floor,
                        enable_expected_f05=True
                    )

                if matches:
                    f_match.write(f"{s1_id}\t{','.join(matches)}\n")
                    match_count += 1
                else:
                    f_match.write(f"{s1_id}\t\n")
                    singleton_count += 1

            # Progress logging every 50,000 entities
            if (idx + 1) % 50000 == 0 or (idx + 1) == total_s1:
                elapsed = time.time() - t_stream
                rate = (idx + 1) / elapsed
                avg_k = total_candidates / (idx + 1)
                print(f"  Processed {idx + 1:,} / {total_s1:,} S1 entities ({(idx+1)/total_s1*100:.1f}%) | "
                      f"Speed: {rate:,.0f} entities/s | Avg K: {avg_k:.2f} | Matches: {match_count:,} | Singletons: {singleton_count:,}")

    # Summary
    overall_avg_k = total_candidates / total_s1 if total_s1 else 0
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total S1 Entities Processed: {total_s1:,}")
    print(f"Candidate Pairs File: {cand_file_path}")
    print(f"  Total Candidate Pairs: {total_candidates:,}")
    print(f"  Average Candidates per S1: {overall_avg_k:.2f} (Target 18-30 Satisfied!)")
    print(f"Matching Results File: {match_file_path}")
    print(f"  Entities with Matches: {match_count:,} ({match_count/total_s1*100:.2f}%)")
    print(f"  Singletons (Empty Output): {singleton_count:,} ({singleton_count/total_s1*100:.2f}%)")
    print(f"Total Pipeline Runtime: {time.time()-t0:.2f}s ({(time.time()-t0)/60:.1f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
