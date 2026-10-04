"""
Diagnostics Step 12: Candidate-But-Rejected True Match Analysis (False Negative Deep-Dive)
========================================================================================
Investigates the 3.62 pp gap between Candidate Recall (98.35%) and Pair Recall (94.73%) at K=30.
Specifically isolates:
    Ground Truth Pair -> Retrieved in Top-30 Candidates -> XGBoost Score < 0.70 (False Negative)

Tests the core architectural hypothesis:
    "Is XGBoost over-relying on address similarity and rejecting genuine matches when the address is noisy or missing?"

Categorizes all False Negatives and computes TP vs FN feature distributions.
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


def compute_pair_features(s1_n, tgt_n, s1_a, tgt_a, s1_c, tgt_c, ret_score):
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


def categorize_false_negative(row):
    """Categorize why a true match inside the candidate set was rejected by XGBoost."""
    # 1. Missing address in one or both records
    if row['both_addr_present'] == 0:
        return 'Missing address'
    
    # 2. Address is weak/noisy but name is strong
    if row['name_jaro_winkler'] >= 0.85 and row['addr_jaccard'] < 0.25:
        return 'Address weak / name strong'
        
    # 3. Name is weak/noisy but address is strong
    if row['addr_jaccard'] >= 0.50 and row['name_jaro_winkler'] < 0.70:
        return 'Name weak / address strong'
        
    # 4. Low retrieval rank / weak initial token overlap
    if row['retrieval_score'] < 0.15:
        return 'Retrieval score too low'
        
    # 5. Severe name divergence despite partial tokens (acronyms / abbreviations)
    s1_w = row['s1_n'].split()
    tgt_w = row['tgt_n'].split()
    if len(s1_w) > 0 and len(tgt_w) > 0:
        s1_acr = "".join([w[0] for w in s1_w if w])
        tgt_acr = "".join([w[0] for w in tgt_w if w])
        if (len(s1_acr) >= 2 and s1_acr in row['tgt_n']) or (len(tgt_acr) >= 2 and tgt_acr in row['s1_n']):
            return 'Acronym / name truncation'
            
    # 6. Both moderately similar (falling in ambiguous zone)
    if 0.65 <= row['name_jaro_winkler'] < 0.85 and 0.20 <= row['addr_jaccard'] < 0.50:
        return 'Both moderately similar'
        
    # 7. Transliteration / Phonetic divergence in name
    if row['name_levenshtein'] < 0.60 and row['name_jaro_winkler'] >= 0.70:
        return 'Transliteration / phonetic variation'
        
    return 'Other residual'


def run_step12():
    print("=" * 80)
    print("STEP 12: CANDIDATE-BUT-REJECTED TRUE MATCH ANALYSIS (XGBoost FALSE NEGATIVES)")
    print("=" * 80)
    
    con = duckdb.connect()
    
    # 1. Load Evaluation Cohort & Ground Truth
    gt_dict_path = os.path.join(REPORTS_DIR, "21_eval_gt_dict.json")
    with open(gt_dict_path, "r", encoding="utf-8") as f:
        gt_dict_raw = json.load(f)
    gt_pair_set = {(s1, tgt) for s1, tgts in gt_dict_raw.items() for tgt in tgts}
    total_eval_gt = len(gt_pair_set)
    print(f"\n[1/5] Loaded {len(gt_dict_raw):,} S1 evaluation entities ({total_eval_gt:,} true pairs).")
    
    # 2. Load Candidates & Filter to K=30
    cand_path = os.path.join(REPORTS_DIR, "21_improved_candidates.parquet")
    df_all_cand = pd.read_parquet(cand_path)
    
    # Sort and rank up to K=30
    df_all_cand = df_all_cand.sort_values(by=['s1_id', 'score'], ascending=[True, False]).reset_index(drop=True)
    df_all_cand['rank'] = df_all_cand.groupby('s1_id').cumcount() + 1
    k30_cand = df_all_cand[df_all_cand['rank'] <= 30].copy().reset_index(drop=True)
    
    retrieved_true_pairs = {(r.s1_id, r.target_id) for r in k30_cand.itertuples(index=False)
                            if (r.s1_id, r.target_id) in gt_pair_set}
    cand_recall = len(retrieved_true_pairs) / total_eval_gt
    print(f"  K=30 Candidate Pool size: {len(k30_cand):,} pairs.")
    print(f"  Retrieved True Pairs: {len(retrieved_true_pairs):,} / {total_eval_gt:,} ({cand_recall*100:.2f}% Candidate Recall)")
    
    # 3. Join with text fields to compute feature vectors
    print("\n[2/5] Joining text data for feature computation...")
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
    
    # Feature extraction
    print("\n[3/5] Vectorizing feature matrix...")
    t0 = time.time()
    pairs_df['s1_n'] = pairs_df['s1_name'].apply(transliterated)
    pairs_df['tgt_n'] = pairs_df['tgt_name'].apply(transliterated)
    pairs_df['s1_a'] = pairs_df['s1_addr'].apply(transliterated)
    pairs_df['tgt_a'] = pairs_df['tgt_addr'].apply(transliterated)
    
    rows = []
    for _, r in pairs_df.iterrows():
        f = compute_pair_features(
            r['s1_n'], r['tgt_n'],
            r['s1_a'], r['tgt_a'],
            r['s1_c'], r['tgt_c'],
            r['ret_score']
        )
        f['s1_id'] = r['s1_id']
        f['target_id'] = r['target_id']
        f['rank'] = r['rank']
        f['country'] = r['s1_c']
        f['s1_n'] = r['s1_n']
        f['tgt_n'] = r['tgt_n']
        f['s1_a'] = r['s1_a']
        f['tgt_a'] = r['tgt_a']
        f['is_match'] = 1 if (r['s1_id'], r['target_id']) in gt_pair_set else 0
        rows.append(f)
        
    df_feat = pd.DataFrame(rows)
    print(f"  Feature extraction completed in {time.time()-t0:.2f}s.")
    
    # 4. Out-of-Fold Model Scoring (3-Fold GroupKFold by s1_id)
    print("\n[4/5] Running Out-of-Fold XGBoost Scoring...")
    feature_cols = [
        'retrieval_score', 'name_exact', 'name_levenshtein', 'name_jaro_winkler', 
        'name_jaccard', 'name_token_overlap', 'name_length_ratio', 'has_addr_s1',
        'has_addr_tgt', 'both_addr_present', 'addr_exact', 'addr_levenshtein', 
        'addr_jaro_winkler', 'addr_jaccard', 'postal_match', 'house_num_match', 
        'name_addr_agree', 'name_postal_agree'
    ]
    
    gkf = GroupKFold(n_splits=3)
    df_feat['oof_prob'] = 0.0
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(df_feat, groups=df_feat['s1_id'])):
        X_tr = df_feat.iloc[trn_idx][feature_cols].values
        y_tr = df_feat.iloc[trn_idx]['is_match'].values
        X_va = df_feat.iloc[val_idx][feature_cols].values
        
        clf = xgb.XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        df_feat.iloc[val_idx, df_feat.columns.get_loc('oof_prob')] = clf.predict_proba(X_va)[:, 1]
        
    # Evaluate at threshold = 0.70
    DECISION_THRESHOLD = 0.70
    df_feat['pred_match'] = (df_feat['oof_prob'] >= DECISION_THRESHOLD).astype(int)
    
    # Isolate True Matches inside candidate pool
    true_candidates = df_feat[df_feat['is_match'] == 1].copy()
    
    # Separate into True Positives (Accepted) vs False Negatives (Rejected)
    tp_df = true_candidates[true_candidates['pred_match'] == 1].copy()
    fn_df = true_candidates[true_candidates['pred_match'] == 0].copy()
    
    n_retrieved_true = len(true_candidates)
    n_tp = len(tp_df)
    n_fn = len(fn_df)
    
    pair_recall_total = n_tp / total_eval_gt
    pair_recall_retrieved = n_tp / n_retrieved_true
    fn_pct_of_true = (n_fn / total_eval_gt) * 100
    fn_pct_of_retrieved = (n_fn / n_retrieved_true) * 100
    
    print("\n" + "=" * 60)
    print(f"CANDIDATE-BUT-REJECTED METRICS SUMMARY")
    print("=" * 60)
    print(f"  Total Ground Truth pairs:            {total_eval_gt:,}")
    print(f"  Retrieved in Candidate Pool (K=30):  {n_retrieved_true:,} ({cand_recall*100:.2f}%)")
    print(f"  Accepted by XGBoost (p >= 0.70) [TP]:{n_tp:,} ({pair_recall_total*100:.2f}% of all GT)")
    print(f"  Rejected by XGBoost (p < 0.70)  [FN]:{n_fn:,} ({fn_pct_of_true:.2f}% of all GT, {fn_pct_of_retrieved:.2f}% of candidate GT)")
    print(f"  The Gap:                             {cand_recall*100 - pair_recall_total*100:.2f} percentage points lost in decision layer!")
    
    # -------------------------------------------------------------
    # 5. Test Hypothesis: Address Over-Reliance & Feature Distributions
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TESTING ADDRESS OVER-RELIANCE HYPOTHESIS (TP vs FN DISTRIBUTIONS)")
    print("=" * 60)
    
    compare_features = [
        'addr_jaccard', 'both_addr_present', 'addr_jaro_winkler', 'addr_levenshtein',
        'name_jaro_winkler', 'name_jaccard', 'name_levenshtein', 'retrieval_score',
        'house_num_match', 'postal_match', 'oof_prob'
    ]
    
    dist_rows = []
    for f in compare_features:
        tp_mean = tp_df[f].mean()
        tp_med = tp_df[f].median()
        fn_mean = fn_df[f].mean()
        fn_med = fn_df[f].median()
        delta_mean = fn_mean - tp_mean
        dist_rows.append({
            'feature': f,
            'TP_mean': round(tp_mean, 4),
            'TP_median': round(tp_med, 4),
            'FN_mean': round(fn_mean, 4),
            'FN_median': round(fn_med, 4),
            'delta_mean (FN - TP)': round(delta_mean, 4)
        })
        print(f"  {f:20s} | TP: mean={tp_mean:.3f}, med={tp_med:.3f} | FN: mean={fn_mean:.3f}, med={fn_med:.3f} | Δ={delta_mean:+.3f}")
        
    df_dist = pd.DataFrame(dist_rows)
    dist_path = os.path.join(REPORTS_DIR, "30_tp_vs_fn_feature_distributions.csv")
    df_dist.to_csv(dist_path, index=False)
    print(f"  --> Saved feature distributions to {dist_path}")
    
    # Check specific conditions on False Negatives:
    fn_high_name_low_addr = len(fn_df[(fn_df['name_jaro_winkler'] >= 0.85) & (fn_df['addr_jaccard'] < 0.25)])
    fn_missing_addr = len(fn_df[fn_df['both_addr_present'] == 0])
    fn_high_retrieval = len(fn_df[fn_df['retrieval_score'] >= 0.50])
    
    print(f"\n  [HYPOTHESIS CHECK]:")
    print(f"    FNs with High Name (JW >= 0.85) & Low Address (Jaccard < 0.25): {fn_high_name_low_addr:,} ({fn_high_name_low_addr/n_fn*100:.1f}%)")
    print(f"    FNs with Missing Address in S1 or Target:                      {fn_missing_addr:,} ({fn_missing_addr/n_fn*100:.1f}%)")
    print(f"    FNs with Strong Retrieval Score (>= 0.50) but rejected:        {fn_high_retrieval:,} ({fn_high_retrieval/n_fn*100:.1f}%)")
    
    # -------------------------------------------------------------
    # 6. Detailed False Negative Categorization
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FALSE NEGATIVE TAXONOMY BREAKDOWN ({n_fn:,} pairs)")
    print("=" * 60)
    
    fn_df['failure_type'] = fn_df.apply(categorize_false_negative, axis=1)
    breakdown = fn_df['failure_type'].value_counts()
    
    summary_table = []
    for ftype, count in breakdown.items():
        pct = (count / n_fn) * 100
        summary_table.append({
            'failure_type': ftype,
            'count': int(count),
            'pct_of_fn': round(pct, 2)
        })
        print(f"  {ftype:35s}: {count:5d} ({pct:5.2f}%)")
        
    df_summary = pd.DataFrame(summary_table)
    summary_path = os.path.join(REPORTS_DIR, "28_rejected_candidates_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)
    print(f"  --> Saved failure taxonomy to {summary_path}")
    
    # Export full detailed breakdown of all False Negatives for error inspection
    fn_export_cols = [
        's1_id', 'target_id', 'country', 'oof_prob', 'failure_type', 'rank',
        'retrieval_score', 'name_jaro_winkler', 'addr_jaccard', 'both_addr_present',
        'postal_match', 'house_num_match', 's1_n', 'tgt_n', 's1_a', 'tgt_a'
    ]
    fn_df_export = fn_df[fn_export_cols].sort_values(by='oof_prob', ascending=False)
    fn_breakdown_path = os.path.join(REPORTS_DIR, "29_rejected_candidates_breakdown.csv")
    fn_df_export.to_csv(fn_breakdown_path, index=False)
    print(f"  --> Saved detailed FN records to {fn_breakdown_path}")
    
    print("\n" + "=" * 80)
    print("STEP 12 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_step12()
