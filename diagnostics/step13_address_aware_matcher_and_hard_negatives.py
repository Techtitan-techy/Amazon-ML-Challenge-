"""
Diagnostics Step 13: Address-Availability-Aware Matcher & Hard-Negative Optimization
====================================================================================
Implements and benchmarks Priority 1 and Priority 2:
1. Conflation Resolution: In standard ER, setting missing address similarity to 0.0
   causes decision trees to treat unobserved addresses as strong negative evidence.
   Here we implement:
   - Native NaN handling in XGBoost for missing address similarities
   - Explicit interaction features:
     * name_strong_addr_missing = (1 - both_addr_present) * name_jaro_winkler
     * name_jaccard_addr_missing = (1 - both_addr_present) * name_jaccard
     * either_addr_missing = 1.0 - both_addr_present
2. Hard-Negative Mining from the actual TF-IDF candidate distribution.
3. Evaluates on 3-Fold GroupKFold to measure:
   - How many of the 1,069 missing-address FNs are recovered
   - Pair Recall (baseline 93.98%)
   - Pair Precision (baseline 98.19%)
   - Macro F0.5 (baseline 0.9617)
"""

import os
import sys
import time
import json
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, REPORTS_DIR
from normalization import transliterated, extract_postal, extract_house_num


def calc_f_beta(precision, recall, beta=0.5):
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom


def calc_macro_f05(pred_df, gt_dict, all_s1_ids):
    preds_by_s1 = pred_df.groupby('s1_id')['target_id'].apply(set).to_dict()
    scores = []
    
    for s1 in all_s1_ids:
        true_set = gt_dict.get(s1, set())
        pred_set = preds_by_s1.get(s1, set())
        
        if len(true_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
        else:
            tp = len(true_set.intersection(pred_set))
            prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
            rec = tp / len(true_set)
            scores.append(calc_f_beta(prec, rec, beta=0.5))
            
    return float(np.mean(scores))


def compute_baseline_features(s1_n, tgt_n, s1_a, tgt_a, s1_c, tgt_c, ret_score):
    """Current feature extractor (zeros for missing address)."""
    n_exact = 1.0 if (s1_n and s1_n == tgt_n) else 0.0
    n_lev = Levenshtein.normalized_similarity(s1_n, tgt_n) if (s1_n and tgt_n) else 0.0
    n_jw = JaroWinkler.similarity(s1_n, tgt_n) if (s1_n and tgt_n) else 0.0
    
    s1_n_toks = set(s1_n.split())
    tgt_n_toks = set(tgt_n.split())
    n_overlap = len(s1_n_toks & tgt_n_toks) / min(len(s1_n_toks), len(tgt_n_toks)) if (s1_n_toks and tgt_n_toks) else 0.0
    n_jaccard = len(s1_n_toks & tgt_n_toks) / len(s1_n_toks | tgt_n_toks) if (s1_n_toks and tgt_n_toks) else 0.0
    n_len_ratio = min(len(s1_n), len(tgt_n)) / max(len(s1_n), len(tgt_n)) if max(len(s1_n), len(tgt_n)) > 0 else 0.0
    
    has_addr_s1 = 1.0 if s1_a else 0.0
    has_addr_tgt = 1.0 if tgt_a else 0.0
    both_addr_present = 1.0 if (has_addr_s1 and has_addr_tgt) else 0.0
    
    a_exact = 1.0 if (both_addr_present and s1_a == tgt_a) else 0.0
    a_lev = Levenshtein.normalized_similarity(s1_a, tgt_a) if both_addr_present else 0.0
    a_jw = JaroWinkler.similarity(s1_a, tgt_a) if both_addr_present else 0.0
    
    s1_a_toks = set(s1_a.split())
    tgt_a_toks = set(tgt_a.split())
    a_jaccard = len(s1_a_toks & tgt_a_toks) / len(s1_a_toks | tgt_a_toks) if (s1_a_toks and tgt_a_toks) else 0.0
    
    p1 = extract_postal(s1_a, s1_c)
    p2 = extract_postal(tgt_a, tgt_c)
    postal_match = 1.0 if (p1 and p2 and p1 == p2) else 0.0
    
    h1 = extract_house_num(s1_a)
    h2 = extract_house_num(tgt_a)
    house_match = 1.0 if (h1 and h2 and h1 == h2) else 0.0
    
    return {
        'retrieval_score': round(float(ret_score), 4),
        'name_exact': n_exact,
        'name_levenshtein': round(n_lev, 4),
        'name_jaro_winkler': round(n_jw, 4),
        'name_jaccard': round(n_jaccard, 4),
        'name_token_overlap': round(n_overlap, 4),
        'name_length_ratio': round(n_len_ratio, 4),
        'has_addr_s1': has_addr_s1,
        'has_addr_tgt': has_addr_tgt,
        'both_addr_present': both_addr_present,
        'addr_exact': a_exact,
        'addr_levenshtein': round(a_lev, 4),
        'addr_jaro_winkler': round(a_jw, 4),
        'addr_jaccard': round(a_jaccard, 4),
        'postal_match': postal_match,
        'house_num_match': house_match,
        'name_addr_agree': 1.0 if (n_exact == 1.0 and a_exact == 1.0) else 0.0,
        'name_postal_agree': 1.0 if (n_exact == 1.0 and postal_match == 1.0) else 0.0,
    }


def compute_address_aware_features(s1_n, tgt_n, s1_a, tgt_a, s1_c, tgt_c, ret_score):
    """
    Address-Availability-Aware feature extractor:
    - Uses np.nan for missing address similarities so XGBoost uses default split path
    - Injects explicit missingness-name interaction features
    """
    f = compute_baseline_features(s1_n, tgt_n, s1_a, tgt_a, s1_c, tgt_c, ret_score)
    both_present = f['both_addr_present']
    
    # 1. Missingness indicator
    either_missing = 1.0 if both_present == 0.0 else 0.0
    f['either_addr_missing'] = either_missing
    
    # 2. Key interactions: when address is missing, how strong is the name?
    # This prevents the tree from treating unobserved addresses as negative proof
    f['name_strong_addr_missing'] = either_missing * f['name_jaro_winkler']
    f['name_jaccard_addr_missing'] = either_missing * f['name_jaccard']
    f['retrieval_x_name'] = round(f['retrieval_score'] * f['name_jaro_winkler'], 4)
    
    # 3. Use np.nan for address similarity features when address is unobserved
    if both_present == 0.0:
        f['addr_jaccard_nan'] = np.nan
        f['addr_jaro_winkler_nan'] = np.nan
        f['addr_levenshtein_nan'] = np.nan
    else:
        f['addr_jaccard_nan'] = f['addr_jaccard']
        f['addr_jaro_winkler_nan'] = f['addr_jaro_winkler']
        f['addr_levenshtein_nan'] = f['addr_levenshtein']
        
    return f


def run_step13():
    print("=" * 80)
    print("STEP 13: ADDRESS-AVAILABILITY-AWARE MATCHER & HARD-NEGATIVE MINING")
    print("=" * 80)
    
    con = duckdb.connect()
    
    # 1. Load Ground Truth Dictionary
    gt_dict_path = os.path.join(REPORTS_DIR, "21_eval_gt_dict.json")
    with open(gt_dict_path, "r", encoding="utf-8") as f:
        gt_dict_raw = json.load(f)
    gt_dict = {s1: set(tgts) for s1, tgts in gt_dict_raw.items()}
    all_s1_eval = list(gt_dict.keys())
    gt_pair_set = {(s1, tgt) for s1, tgts in gt_dict.items() for tgt in tgts}
    total_eval_gt = len(gt_pair_set)
    print(f"\n[1/5] Loaded {len(all_s1_eval):,} S1 evaluation entities ({total_eval_gt:,} true pairs).")
    
    # 2. Load Candidates & Filter to K=30
    cand_path = os.path.join(REPORTS_DIR, "21_improved_candidates.parquet")
    df_all_cand = pd.read_parquet(cand_path)
    df_all_cand = df_all_cand.sort_values(by=['s1_id', 'score'], ascending=[True, False]).reset_index(drop=True)
    df_all_cand['rank'] = df_all_cand.groupby('s1_id').cumcount() + 1
    k30_cand = df_all_cand[df_all_cand['rank'] <= 30].copy().reset_index(drop=True)
    
    # 3. Join with text fields
    print("\n[2/5] Joining candidate pairs with source texts...")
    t0 = time.time()
    con.register('df_cand_k30', k30_cand)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE targets_pool AS
        SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True)
        UNION ALL
        SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True)
    """)
    
    pairs_df = con.execute(f"""
        SELECT 
            c.s1_id, c.target_id, c.score as ret_score, c.rank,
            s1.business_name as s1_name, s1.business_address as s1_addr, s1.country as s1_c,
            t.business_name as tgt_name, t.business_address as tgt_addr, t.country as tgt_c
        FROM df_cand_k30 c
        JOIN read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True) s1 ON c.s1_id = s1.entity_id
        JOIN targets_pool t ON c.target_id = t.entity_id
    """).fetch_df()
    print(f"  Joined {len(pairs_df):,} pairs in {time.time()-t0:.2f}s.")
    
    # 4. Compute Features for Both Baseline and Address-Aware
    print("\n[3/5] Vectorizing Baseline and Address-Aware feature sets...")
    t0 = time.time()
    pairs_df['s1_n'] = pairs_df['s1_name'].apply(transliterated)
    pairs_df['tgt_n'] = pairs_df['tgt_name'].apply(transliterated)
    pairs_df['s1_a'] = pairs_df['s1_addr'].apply(transliterated)
    pairs_df['tgt_a'] = pairs_df['tgt_addr'].apply(transliterated)
    
    feat_rows = []
    for _, r in pairs_df.iterrows():
        f = compute_address_aware_features(
            r['s1_n'], r['tgt_n'],
            r['s1_a'], r['tgt_a'],
            r['s1_c'], r['tgt_c'],
            r['ret_score']
        )
        f['s1_id'] = r['s1_id']
        f['target_id'] = r['target_id']
        f['is_match'] = 1 if (r['s1_id'], r['target_id']) in gt_pair_set else 0
        feat_rows.append(f)
        
    df_feat = pd.DataFrame(feat_rows)
    print(f"  Feature extraction finished ({len(df_feat):,} pairs) in {time.time()-t0:.2f}s.")
    
    # Define Feature Sets
    baseline_cols = [
        'retrieval_score', 'name_exact', 'name_levenshtein', 'name_jaro_winkler', 
        'name_jaccard', 'name_token_overlap', 'name_length_ratio', 'has_addr_s1',
        'has_addr_tgt', 'both_addr_present', 'addr_exact', 'addr_levenshtein', 
        'addr_jaro_winkler', 'addr_jaccard', 'postal_match', 'house_num_match', 
        'name_addr_agree', 'name_postal_agree'
    ]
    
    address_aware_cols = [
        'retrieval_score', 'name_exact', 'name_levenshtein', 'name_jaro_winkler', 
        'name_jaccard', 'name_token_overlap', 'name_length_ratio', 'has_addr_s1',
        'has_addr_tgt', 'both_addr_present', 'either_addr_missing',
        'addr_exact', 'addr_levenshtein_nan', 'addr_jaro_winkler_nan', 'addr_jaccard_nan', 
        'postal_match', 'house_num_match', 'name_addr_agree', 'name_postal_agree',
        'name_strong_addr_missing', 'name_jaccard_addr_missing', 'retrieval_x_name'
    ]
    
    # 5. Run 3-Fold GroupKFold Cross-Validation Benchmark
    print("\n[4/5] Running 3-Fold GroupKFold Cross-Validation on Held-Out S1 Entities...")
    gkf = GroupKFold(n_splits=3)
    
    df_feat['oof_prob_baseline'] = 0.0
    df_feat['oof_prob_addr_aware'] = 0.0
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(df_feat, groups=df_feat['s1_id'])):
        print(f"  --- Fold {fold+1}/3 ---")
        y_tr = df_feat.iloc[trn_idx]['is_match'].values
        
        # 1. Baseline Model
        X_tr_base = df_feat.iloc[trn_idx][baseline_cols].values
        X_va_base = df_feat.iloc[val_idx][baseline_cols].values
        clf_base = xgb.XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
        clf_base.fit(X_tr_base, y_tr)
        df_feat.iloc[val_idx, df_feat.columns.get_loc('oof_prob_baseline')] = clf_base.predict_proba(X_va_base)[:, 1]
        
        # 2. Address-Aware Model (Native NaN + Interaction Features)
        X_tr_aware = df_feat.iloc[trn_idx][address_aware_cols].values
        X_va_aware = df_feat.iloc[val_idx][address_aware_cols].values
        clf_aware = xgb.XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
        clf_aware.fit(X_tr_aware, y_tr)
        df_feat.iloc[val_idx, df_feat.columns.get_loc('oof_prob_addr_aware')] = clf_aware.predict_proba(X_va_aware)[:, 1]

    # 6. Comparative Evaluation
    print("\n[5/5] Evaluating Macro F0.5 & False Negative Recovery...")
    
    results = []
    
    # Config A: Baseline (flat threshold = 0.70)
    pred_base = df_feat[df_feat['oof_prob_baseline'] >= 0.70][['s1_id', 'target_id']].copy()
    tp_base = len(set(zip(pred_base['s1_id'], pred_base['target_id'])) & gt_pair_set)
    fp_base = len(pred_base) - tp_base
    prec_base = tp_base / len(pred_base) if len(pred_base) > 0 else 0.0
    rec_base = tp_base / total_eval_gt
    macro_base = calc_macro_f05(pred_base, gt_dict, all_s1_eval)
    
    results.append({
        'configuration': '1. Baseline XGBoost (Flat 0.70)',
        'pair_precision': round(prec_base, 4),
        'pair_recall': round(rec_base, 4),
        'macro_f05': round(macro_base, 5),
        'accepted_pairs': len(pred_base),
        'true_positives': tp_base,
        'false_positives': fp_base
    })
    
    # Config B: Address-Aware Model (Flat 0.70)
    pred_aware_flat = df_feat[df_feat['oof_prob_addr_aware'] >= 0.70][['s1_id', 'target_id']].copy()
    tp_aware_flat = len(set(zip(pred_aware_flat['s1_id'], pred_aware_flat['target_id'])) & gt_pair_set)
    fp_aware_flat = len(pred_aware_flat) - tp_aware_flat
    prec_aware_flat = tp_aware_flat / len(pred_aware_flat) if len(pred_aware_flat) > 0 else 0.0
    rec_aware_flat = tp_aware_flat / total_eval_gt
    macro_aware_flat = calc_macro_f05(pred_aware_flat, gt_dict, all_s1_eval)
    
    results.append({
        'configuration': '2. Address-Aware Model (Flat 0.70)',
        'pair_precision': round(prec_aware_flat, 4),
        'pair_recall': round(rec_aware_flat, 4),
        'macro_f05': round(macro_aware_flat, 5),
        'accepted_pairs': len(pred_aware_flat),
        'true_positives': tp_aware_flat,
        'false_positives': fp_aware_flat
    })
    
    # Config C: Address-Aware Model + Calibrated Missing-Address Threshold
    # Condition: If address is missing, threshold = 0.50 (if name_jaro >= 0.88), else 0.70
    cond_missing_accept = (
        (df_feat['both_addr_present'] == 0.0) & 
        (df_feat['name_jaro_winkler'] >= 0.88) & 
        (df_feat['retrieval_score'] >= 0.60) &
        (df_feat['oof_prob_addr_aware'] >= 0.50)
    )
    cond_standard_accept = (df_feat['oof_prob_addr_aware'] >= 0.70)
    
    pred_aware_calib = df_feat[cond_standard_accept | cond_missing_accept][['s1_id', 'target_id']].copy()
    tp_aware_calib = len(set(zip(pred_aware_calib['s1_id'], pred_aware_calib['target_id'])) & gt_pair_set)
    fp_aware_calib = len(pred_aware_calib) - tp_aware_calib
    prec_aware_calib = tp_aware_calib / len(pred_aware_calib) if len(pred_aware_calib) > 0 else 0.0
    rec_aware_calib = tp_aware_calib / total_eval_gt
    macro_aware_calib = calc_macro_f05(pred_aware_calib, gt_dict, all_s1_eval)
    
    results.append({
        'configuration': '3. Address-Aware + Missingness Calibration',
        'pair_precision': round(prec_aware_calib, 4),
        'pair_recall': round(rec_aware_calib, 4),
        'macro_f05': round(macro_aware_calib, 5),
        'accepted_pairs': len(pred_aware_calib),
        'true_positives': tp_aware_calib,
        'false_positives': fp_aware_calib
    })
    
    # Config D: Fine Threshold Search on Address-Aware Model
    for t_val in [0.65, 0.60]:
        pred_t = df_feat[df_feat['oof_prob_addr_aware'] >= t_val][['s1_id', 'target_id']].copy()
        tp_t = len(set(zip(pred_t['s1_id'], pred_t['target_id'])) & gt_pair_set)
        fp_t = len(pred_t) - tp_t
        prec_t = tp_t / len(pred_t) if len(pred_t) > 0 else 0.0
        rec_t = tp_t / total_eval_gt
        macro_t = calc_macro_f05(pred_t, gt_dict, all_s1_eval)
        results.append({
            'configuration': f'4. Address-Aware (Threshold {t_val:.2f})',
            'pair_precision': round(prec_t, 4),
            'pair_recall': round(rec_t, 4),
            'macro_f05': round(macro_t, 5),
            'accepted_pairs': len(pred_t),
            'true_positives': tp_t,
            'false_positives': fp_t
        })
        
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("END-TO-END BENCHMARK RESULTS")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    results_path = os.path.join(REPORTS_DIR, "31_address_aware_benchmark.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\n  --> Saved benchmark results to {results_path}")
    
    # 7. Check specific recovery of the 1,069 missing-address FNs
    print("\n" + "=" * 80)
    print("RECOVERY OF PREVIOUSLY REJECTED FALSE NEGATIVES")
    print("=" * 80)
    
    fn_path = os.path.join(REPORTS_DIR, "29_rejected_candidates_breakdown.csv")
    if os.path.exists(fn_path):
        prev_fn_df = pd.read_csv(fn_path)
        prev_fn_set = set(zip(prev_fn_df['s1_id'], prev_fn_df['target_id']))
        prev_missing_set = set(zip(
            prev_fn_df[prev_fn_df['failure_type'] == 'Missing address']['s1_id'],
            prev_fn_df[prev_fn_df['failure_type'] == 'Missing address']['target_id']
        ))
        
        # Check how many were recovered by Config 3
        calib_pred_set = set(zip(pred_aware_calib['s1_id'], pred_aware_calib['target_id']))
        recov_all_fn = len(calib_pred_set & prev_fn_set)
        recov_missing_fn = len(calib_pred_set & prev_missing_set)
        
        print(f"  Total Previous False Negatives:             {len(prev_fn_set):,}")
        print(f"  Recovered by Address-Aware Calibrated Model:{recov_all_fn:,} / {len(prev_fn_set):,} ({recov_all_fn/len(prev_fn_set)*100:.1f}%)")
        print(f"  Specifically Missing-Address FNs Recovered: {recov_missing_fn:,} / {len(prev_missing_set):,} ({recov_missing_fn/len(prev_missing_set)*100:.1f}%)")
        
        recov_summary = {
            'total_previous_fn': len(prev_fn_set),
            'recovered_fn_total': recov_all_fn,
            'recovered_fn_pct': round(recov_all_fn / len(prev_fn_set) * 100, 2),
            'total_missing_address_fn': len(prev_missing_set),
            'recovered_missing_address_fn': recov_missing_fn,
            'recovered_missing_address_pct': round(recov_missing_fn / len(prev_missing_set) * 100, 2),
        }
        recov_path = os.path.join(REPORTS_DIR, "32_fn_recovery_summary.json")
        with open(recov_path, "w", encoding="utf-8") as f:
            json.dump(recov_summary, f, indent=2)
        print(f"  --> Saved recovery summary to {recov_path}")


if __name__ == "__main__":
    run_step13()
