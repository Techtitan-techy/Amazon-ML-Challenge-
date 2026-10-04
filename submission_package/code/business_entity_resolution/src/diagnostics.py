"""
Championship Diagnostics & Segment-Specific Scoreboard.
Amazon ML Challenge 2026.

Implements:
- Stage 17: Segment-Specific Scoreboard (Overall, US, India, S2, S3, Singletons, Multi-match, Missing-Address)
- Stage 22: Oracle Candidate Diagnostics (Recall@K and Theoretical Max F0.5 ceiling)
"""

from typing import Dict, List, Set, Any, Tuple
import numpy as np
import pandas as pd


def compute_entity_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Tuple[float, float, float]:
    """
    Calculate the official Macro F0.5 score across all S1 entities:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons score 1.0 when prediction is empty, and 0.0 when prediction is non-empty.
    """
    all_s1 = set(ground_truth.keys()) | set(predictions.keys())
    if not all_s1:
        return 0.0, 0.0, 0.0

    entity_f05_scores = []
    entity_precisions = []
    entity_recalls = []

    for s1 in all_s1:
        true_set = ground_truth.get(s1, set())
        pred_set = predictions.get(s1, set())

        # Singleton evaluation
        if len(true_set) == 0:
            if len(pred_set) == 0:
                entity_f05_scores.append(1.0)
                entity_precisions.append(1.0)
                entity_recalls.append(1.0)
            else:
                entity_f05_scores.append(0.0)
                entity_precisions.append(0.0)
                entity_recalls.append(1.0)
            continue

        # Non-singleton evaluation
        if len(pred_set) == 0:
            entity_f05_scores.append(0.0)
            entity_precisions.append(1.0)  # No false positives made
            entity_recalls.append(0.0)
            continue

        tp = len(true_set & pred_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)

        denom = 0.25 * p + r
        f05 = (1.25 * p * r) / denom if denom > 0 else 0.0

        entity_f05_scores.append(f05)
        entity_precisions.append(p)
        entity_recalls.append(r)

    macro_f05 = float(np.mean(entity_f05_scores))
    macro_p = float(np.mean(entity_precisions))
    macro_r = float(np.mean(entity_recalls))

    return macro_f05, macro_p, macro_r


def run_oracle_candidate_diagnostics(
    ground_truth: Dict[str, Set[str]],
    candidates_by_s1: Dict[str, List[str]],
    k_cutoffs: List[int] = [10, 20, 25, 30, 35, 40]
) -> pd.DataFrame:
    """
    Stage 22: Oracle Candidate Diagnostics.
    If every true candidate inside the candidate pool were perfectly classified,
    what Macro F0.5 is theoretically achievable?
    Separates retrieval limits from classifier limits.
    """
    results = []

    for k in k_cutoffs:
        oracle_preds = {}
        total_true_links = 0
        recalled_links = 0

        for s1, true_set in ground_truth.items():
            pool = set(candidates_by_s1.get(s1, [])[:k])
            # Oracle matches = true links present in candidate pool
            oracle_matches = true_set & pool
            oracle_preds[s1] = oracle_matches

            total_true_links += len(true_set)
            recalled_links += len(oracle_matches)

        macro_f05, macro_p, macro_r = compute_entity_macro_f05(ground_truth, oracle_preds)
        cand_recall = recalled_links / total_true_links if total_true_links > 0 else 0.0

        results.append({
            "K_Cutoff": k,
            "Candidate_Recall": cand_recall,
            "Oracle_Macro_F0.5": macro_f05,
            "Oracle_Precision": macro_p,
            "Oracle_Recall": macro_r
        })

    return pd.DataFrame(results)


def generate_segment_scoreboard(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_metadata: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Stage 17: Segment-Specific Scoreboard.
    Evaluates Macro F0.5 across key competition slices.
    """
    segments = {
        "Overall": list(ground_truth.keys()),
        "US": [s1 for s1, m in s1_metadata.items() if m.get("country") == "US" and s1 in ground_truth],
        "India": [s1 for s1, m in s1_metadata.items() if m.get("country") == "INDIA" and s1 in ground_truth],
        "Singletons (0-match)": [s1 for s1, true_set in ground_truth.items() if len(true_set) == 0],
        "1-match Entities": [s1 for s1, true_set in ground_truth.items() if len(true_set) == 1],
        "Multi-match Entities (2+)": [s1 for s1, true_set in ground_truth.items() if len(true_set) >= 2],
        "Missing Address S1": [s1 for s1, m in s1_metadata.items() if not m.get("norm_addr") and s1 in ground_truth],
        "Address Present S1": [s1 for s1, m in s1_metadata.items() if m.get("norm_addr") and s1 in ground_truth],
    }

    rows = []
    for seg_name, s1_ids in segments.items():
        if not s1_ids:
            continue
        seg_gt = {s1: ground_truth.get(s1, set()) for s1 in s1_ids}
        seg_pred = {s1: predictions.get(s1, set()) for s1 in s1_ids}

        f05, p, r = compute_entity_macro_f05(seg_gt, seg_pred)
        rows.append({
            "Segment": seg_name,
            "Count": len(s1_ids),
            "Macro_F0.5": f"{f05:.4f}",
            "Precision": f"{p:.4f}",
            "Recall": f"{r:.4f}"
        })

    return pd.DataFrame(rows)
