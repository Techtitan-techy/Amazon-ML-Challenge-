"""
Diagnostics Step 14: Adaptive K Candidate Pruning Evaluation
=============================================================
Evaluates Priority 3: Adaptive K vs Globally Fixed K=30.
Tests whether dynamic candidate budgeting:
  - Relative score cutoff (e.g., score >= 0.25 * top_score)
  - Score gap stopping (e.g., if top-1 score >= 0.90 and gap to next > 0.40)
  - Minimum floor K_min = 3, Maximum ceiling K_max = 30
can drastically reduce candidate volume from 30 cands/S1 down to 8-15 cands/S1
while preserving >= 98.3% candidate recall and >= 0.963 Macro F0.5.
"""

import os
import sys
import time
import json
import duckdb
import numpy as np
import pandas as pd
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REPORTS_DIR

def run_step14():
    print("=" * 80)
    print("STEP 14: ADAPTIVE K CANDIDATE PRUNING BENCHMARK")
    print("=" * 80)
    
    # 1. Load Ground Truth Dictionary
    gt_dict_path = os.path.join(REPORTS_DIR, "21_eval_gt_dict.json")
    with open(gt_dict_path, "r", encoding="utf-8") as f:
        gt_dict_raw = json.load(f)
    gt_dict = {s1: set(tgts) for s1, tgts in gt_dict_raw.items()}
    all_s1_eval = list(gt_dict.keys())
    gt_pair_set = {(s1, tgt) for s1, tgts in gt_dict.items() for tgt in tgts}
    total_eval_gt = len(gt_pair_set)
    print(f"\n[1/3] Loaded {len(all_s1_eval):,} S1 evaluation entities ({total_eval_gt:,} true pairs).")
    
    # 2. Load Full Retrieved Candidate Pool up to K=50
    cand_path = os.path.join(REPORTS_DIR, "21_improved_candidates.parquet")
    df_all_cand = pd.read_parquet(cand_path)
    df_all_cand = df_all_cand.sort_values(by=['s1_id', 'score'], ascending=[True, False]).reset_index(drop=True)
    df_all_cand['rank'] = df_all_cand.groupby('s1_id').cumcount() + 1
    
    # Baseline Fixed K=30
    k30_cand = df_all_cand[df_all_cand['rank'] <= 30].copy().reset_index(drop=True)
    k30_pairs = set(zip(k30_cand['s1_id'], k30_cand['target_id']))
    hits_k30 = len(k30_pairs & gt_pair_set)
    rec_k30 = hits_k30 / total_eval_gt
    
    print("\n" + "=" * 80)
    print("BASELINE FIXED K CONFIGURATIONS")
    print("=" * 80)
    print(f"  Fixed K=10: Candidates = {len(df_all_cand[df_all_cand['rank'] <= 10]):,} | "
          f"Recall = {len(set(zip(df_all_cand[df_all_cand['rank'] <= 10]['s1_id'], df_all_cand[df_all_cand['rank'] <= 10]['target_id'])) & gt_pair_set) / total_eval_gt * 100:.2f}%")
    print(f"  Fixed K=20: Candidates = {len(df_all_cand[df_all_cand['rank'] <= 20]):,} | "
          f"Recall = {len(set(zip(df_all_cand[df_all_cand['rank'] <= 20]['s1_id'], df_all_cand[df_all_cand['rank'] <= 20]['target_id'])) & gt_pair_set) / total_eval_gt * 100:.2f}%")
    print(f"  Fixed K=30: Candidates = {len(k30_cand):,} | Recall = {rec_k30 * 100:.2f}%")
    
    # 3. Test Adaptive K Policies
    print("\n" + "=" * 80)
    print("TESTING ADAPTIVE K POLICIES")
    print("=" * 80)
    
    # Group candidates by s1_id
    grouped_cands = df_all_cand[df_all_cand['rank'] <= 30].groupby('s1_id')[['target_id', 'score', 'rank']].apply(
        lambda g: list(zip(g['target_id'], g['score'], g['rank']))
    ).to_dict()
    
    policies = [
        # Relative ratio cutoff: keep candidates with score >= ratio * top_score, up to max_k
        {'name': 'Adaptive Ratio 0.20 (min=3, max=30)', 'ratio': 0.20, 'min_k': 3, 'max_k': 30, 'steep_cut': False},
        {'name': 'Adaptive Ratio 0.25 (min=3, max=30)', 'ratio': 0.25, 'min_k': 3, 'max_k': 30, 'steep_cut': False},
        {'name': 'Adaptive Ratio 0.30 (min=3, max=30)', 'ratio': 0.30, 'min_k': 3, 'max_k': 30, 'steep_cut': False},
        {'name': 'Adaptive Ratio 0.35 (min=3, max=30)', 'ratio': 0.35, 'min_k': 3, 'max_k': 30, 'steep_cut': False},
        {'name': 'Adaptive Steep Dropoff (top >= 0.85 & gap > 0.40 -> K=3, else 30)', 'ratio': 0.0, 'min_k': 3, 'max_k': 30, 'steep_cut': True},
        {'name': 'Adaptive Hybrid (Ratio 0.25 + Steep Dropoff)', 'ratio': 0.25, 'min_k': 3, 'max_k': 30, 'steep_cut': True},
    ]
    
    results = []
    results.append({
        'policy': 'Fixed K=30 (Baseline)',
        'candidate_recall': round(rec_k30 * 100, 2),
        'total_candidates': len(k30_cand),
        'avg_candidates_per_s1': round(len(k30_cand) / len(all_s1_eval), 2),
        'candidate_reduction_pct': 0.0,
        'true_hits': hits_k30
    })
    
    for pol in policies:
        selected_pairs = set()
        
        for s1_id in all_s1_eval:
            cands = grouped_cands.get(s1_id, [])
            if not cands:
                continue
                
            top_score = cands[0][1]
            
            # Apply policy
            chosen = []
            for tid, sc, rk in cands:
                # Always take up to min_k
                if rk <= pol['min_k']:
                    chosen.append((s1_id, tid))
                    continue
                    
                # Hard ceiling max_k
                if rk > pol['max_k']:
                    break
                    
                # Steep dropoff condition
                if pol['steep_cut'] and top_score >= 0.85 and (top_score - sc) >= 0.40 and rk > 3:
                    break
                    
                # Ratio condition
                if pol['ratio'] > 0.0 and sc < (pol['ratio'] * top_score) and rk > pol['min_k']:
                    break
                    
                chosen.append((s1_id, tid))
                
            selected_pairs.update(chosen)
            
        hits = len(selected_pairs & gt_pair_set)
        rec = hits / total_eval_gt
        reduction = (1.0 - (len(selected_pairs) / len(k30_cand))) * 100
        
        results.append({
            'policy': pol['name'],
            'candidate_recall': round(rec * 100, 2),
            'total_candidates': len(selected_pairs),
            'avg_candidates_per_s1': round(len(selected_pairs) / len(all_s1_eval), 2),
            'candidate_reduction_pct': round(reduction, 2),
            'true_hits': hits
        })
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
    out_path = os.path.join(REPORTS_DIR, "33_adaptive_k_benchmark.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\n  --> Saved Adaptive K benchmark to {out_path}")


if __name__ == "__main__":
    run_step14()
