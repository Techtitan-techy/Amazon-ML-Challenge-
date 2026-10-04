"""
Training Script for Production Matchers & Championship Ensemble.
Amazon ML Challenge 2026.

Supports:
- Training standard XGBoost Matcher with Hard Negatives
- Training Diverse Ensemble Matcher (XGBoost + LightGBM + HistGradientBoosting) with Platt Calibration
- Integration with HardNegativeMiner (H1-H4 error categories)
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import joblib

# Internal modules
from normalize import normalize_record
from features import extract_pair_features, extract_championship_features
from model import create_xgboost_matcher, EnsembleMatcher
from hard_negatives import HardNegativeMiner


def train_production_matcher(
    train_dir: Path,
    output_model_path: Path,
    max_entities: int = 6000,
    use_ensemble: bool = False
):
    """
    Train production XGBoost matcher or championship diverse ensemble.
    
    Args:
        train_dir: Path to directory containing train source and ground truth TSV files.
        output_model_path: Path where the trained joblib model will be saved.
        max_entities: Maximum number of Source 1 entities to sample for training.
        use_ensemble: Whether to train diverse ensemble (XGBoost + LightGBM + HistGradientBoosting).
    """
    print("=" * 80)
    print(f"TRAINING {'CHAMPIONSHIP DIVERSE ENSEMBLE' if use_ensemble else 'PRODUCTION XGBOOST MATCHER'}")
    print("=" * 80)

    t0 = time.time()
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    print("Loading Ground Truth...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    s1_needed = set()
    s2_needed = set()
    s3_needed = set()
    pos_pairs = []

    for idx, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        matches = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
        if matches and len(s1_needed) < max_entities:
            s1_needed.add(s1)
            for m in matches:
                pos_pairs.append((s1, m, 1))
                if m.startswith("S2-"):
                    s2_needed.add(m)
                elif m.startswith("S3-"):
                    s3_needed.add(m)
        if len(s1_needed) >= max_entities:
            break

    print(f"Sampled {len(s1_needed):,} S1 entities | True links to fetch: {len(pos_pairs):,}")

    # Load S1 records
    print("Loading Source 1 records...")
    s1_dict = {}
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=300000):
        mc = chunk[chunk["entity_id"].isin(s1_needed)]
        for _, r in mc.iterrows():
            s1_dict[r["entity_id"]] = normalize_record(r.to_dict())
        if len(s1_dict) == len(s1_needed):
            break

    # Load S2 and S3 records
    print("Loading Target records (S2 and S3)...")
    target_dict = {}
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False, chunksize=300000):
        mc = chunk[chunk["entity_id"].isin(s2_needed)]
        for _, r in mc.iterrows():
            target_dict[r["entity_id"]] = normalize_record(r.to_dict())
        if len(target_dict) == len(s2_needed):
            break

    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False, chunksize=300000):
        mc = chunk[chunk["entity_id"].isin(s3_needed)]
        for _, r in mc.iterrows():
            target_dict[r["entity_id"]] = normalize_record(r.to_dict())
        if len(target_dict) == (len(s2_needed) + len(s3_needed)):
            break

    # Valid positive pairs
    valid_positives = [
        (s1, m, 1) for s1, m, _ in pos_pairs
        if s1 in s1_dict and m in target_dict
    ]
    gt_map: Dict[str, Set[str]] = {}
    for s1, m, _ in valid_positives:
        gt_map.setdefault(s1, set()).add(m)

    print(f"Loaded {len(valid_positives):,} verified positive pairs in {time.time()-t0:.2f}s")

    # Mine Hard Negatives
    print("Mining hard negative candidate pairs across H1-H4 channels...")
    miner = HardNegativeMiner(random_state=42)
    hard_negatives = miner.mine_hard_negatives(
        s1_records=s1_dict,
        target_records=target_dict,
        ground_truth=gt_map,
        max_neg_per_pos=4
    )

    # Sample random distractor negatives to establish global non-match baseline
    print("Sampling random distractor negatives...")
    random_negatives = []
    target_ids_list = list(target_dict.keys())
    rng = np.random.RandomState(42)
    for s1 in s1_dict:
        true_set = gt_map.get(s1, set())
        for _ in range(2):
            rand_tid = target_ids_list[rng.randint(0, len(target_ids_list))]
            if rand_tid not in true_set:
                random_negatives.append((s1, rand_tid, 0))

    dataset_pairs = valid_positives + hard_negatives + random_negatives
    np.random.shuffle(dataset_pairs)
    print(f"Total training pairs: {len(dataset_pairs):,} (Pos: {len(valid_positives):,}, HardNeg: {len(hard_negatives):,}, RandNeg: {len(random_negatives):,})")

    # Extract features
    print("Extracting pairwise features...")
    X_rows = []
    y_rows = []
    for s1, target_id, label in dataset_pairs:
        s1_rec = s1_dict[s1]
        target_rec = target_dict[target_id]
        X_rows.append(extract_pair_features(s1_rec, target_rec))
        y_rows.append(label)

    X_train = np.array(X_rows, dtype=np.float32)
    y_train = np.array(y_rows, dtype=np.int32)

    # Fit Model
    if use_ensemble:
        print("Fitting Championship Diverse Ensemble (XGB + LGB + HGB) with Platt Calibration...")
        model = EnsembleMatcher(xgb_weight=0.45, lgb_weight=0.40, hgb_weight=0.15, calibrate=True)
        model.fit(X_train, y_train)
    else:
        print("Fitting Production XGBoost Classifier (max_depth=6, lr=0.08, n_estimators=300)...")
        model = create_xgboost_matcher()
        model.fit(X_train, y_train)

    # Save model
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(output_model_path))
    print(f"Successfully saved trained model to {output_model_path} in {time.time()-t0:.2f}s!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", action="store_true", help="Train Championship Diverse Ensemble")
    parser.add_argument("--max-entities", type=int, default=6000, help="Max entities to sample")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    train_dir = PROJECT_ROOT / "student_resource" / "dataset" / "train"
    model_name = "ensemble_matcher.joblib" if args.ensemble else "xgboost_matcher.joblib"
    model_output = Path(__file__).resolve().parent / "models" / model_name

    train_production_matcher(train_dir, model_output, max_entities=args.max_entities, use_ensemble=args.ensemble)
