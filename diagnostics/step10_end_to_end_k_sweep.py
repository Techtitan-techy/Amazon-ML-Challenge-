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
from sklearn.metrics import precision_score, recall_score, average_precision_score

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

def compute_pair_features(s1_n, tgt_n, s1_a, tgt_a, s1_c, tgt_c, ret_score):
    n_exact = 1.0 if (s1_n and s1_n == tgt_n) else 0.0
    n_lev = Levenshtein.normalized_similarity(s1_n, tgt_n) if (s1_n and tgt_n) else 0.0
    n_jw = JaroWinkler.similarity(s1_n, tgt_n) if (s1_n and tgt_n) else 0.0
    
    s1_n_toks = set(s1_n.split())
    tgt_n_toks = set(tgt_n.split())
    n_jaccard = len(s1_n_toks & tgt_n_toks) / len(s1_n_toks | tgt_n_toks) if (s1_n_toks and tgt_n_toks) else 0.0
    n_overlap = len(s1_n_toks & tgt_n_toks) / min(len(s1_n_toks), len(tgt_n_toks)) if (s1_n_toks and tgt_n_toks) else 0.0
    n_len_ratio = min(len(s1_n), len(tgt_n)) / max(len(s1_n), len(tgt_n)) if max(len(s1_n), len(tgt_n)) > 0 else 0.0
    
    both_addr_present = 1.0 if (s1_a and tgt_a) else 0.0
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

def run_step10():
    print("=" * 75)
    print("DECISIVE EXPERIMENT: END-TO-END K-SWEEP (K=20, 30, 40, 50) ON MACRO F0.5")
    print("=" * 75)
    
    con = duckdb.connect()
    
    # 1. Load the exact evaluation ground truth dictionary from step07
    print("\n[1/4] Loading evaluation cohort and Ground Truth from step07...")
    gt_dict_path = os.path.join(REPORTS_DIR, "21_eval_gt_dict.json")
    if not os.path.exists(gt_dict_path):
        print(f"Error: {gt_dict_path} missing. Run step07 first.")
        return
        
    with open(gt_dict_path, "r", encoding="utf-8") as f:
        gt_dict_raw = json.load(f)
        
    all_s1_eval = list(gt_dict_raw.keys())
    total_s1 = len(all_s1_eval)
    gt_dict = {s1: set(tgts) for s1, tgts in gt_dict_raw.items()}
    gt_pair_set = {(s1, tgt) for s1, tgts in gt_dict_raw.items() for tgt in tgts}
    total_eval_gt = len(gt_pair_set)
    print(f"  Loaded {total_s1:,} S1 evaluation entities with {total_eval_gt:,} true Ground Truth pairs.")
    
    # 2. Check if improved candidates exist or load scored candidates
    cand_path = os.path.join(REPORTS_DIR, "21_improved_candidates.parquet")
    if not os.path.exists(cand_path):
        print(f"Error: {cand_path} missing. Run step07 first.")
        return
        
    print("\n[2/4] Loading candidate pool and preparing text joins...")
    t0 = time.time()
    df_all_cand = pd.read_parquet(cand_path)
    
    # Join with texts
    con.register('df_cand', df_all_cand)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE targets_all AS
        SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True)
        UNION ALL
        SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True)
    """)
    
    pairs_with_text = con.execute(f"""
        SELECT 
            c.s1_id, c.target_id, c.score as ret_score,
            s1.business_name as s1_name, s1.business_address as s1_addr, s1.country as s1_c,
            t.business_name as tgt_name, t.business_address as tgt_addr, t.country as tgt_c
        FROM df_cand c
        JOIN read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True) s1 ON c.s1_id = s1.entity_id
        JOIN targets_all t ON c.target_id = t.entity_id
    """).fetch_df()
    
    print(f"  Joined {len(pairs_with_text):,} pairs in {time.time()-t0:.2f}s.")
    
    # Compute features for all pairs in pool
    print("\n[3/4] Vectorizing feature matrix for candidate pool...")
    t0 = time.time()
    
    # Pre-normalize for speed
    pairs_with_text['s1_n'] = pairs_with_text['s1_name'].apply(transliterated)
    pairs_with_text['tgt_n'] = pairs_with_text['tgt_name'].apply(transliterated)
    pairs_with_text['s1_a'] = pairs_with_text['s1_addr'].apply(transliterated)
    pairs_with_text['tgt_a'] = pairs_with_text['tgt_addr'].apply(transliterated)
    
    feature_rows = []
    for _, r in pairs_with_text.iterrows():
        f = compute_pair_features(
            r['s1_n'], r['tgt_n'],
            r['s1_a'], r['tgt_a'],
            r['s1_c'], r['tgt_c'],
            r['ret_score']
        )
        f['s1_id'] = r['s1_id']
        f['target_id'] = r['target_id']
        f['is_match'] = 1 if (r['s1_id'], r['target_id']) in gt_pair_set else 0
        feature_rows.append(f)
        
    df_features = pd.DataFrame(feature_rows)
    print(f"  Computed feature matrix ({len(df_features):,} rows) in {time.time()-t0:.2f}s.")
    
    # Compute rank per S1 entity
    df_features = df_features.sort_values(by=['s1_id', 'retrieval_score'], ascending=[True, False]).reset_index(drop=True)
    df_features['rank'] = df_features.groupby('s1_id').cumcount() + 1
    
    feature_cols = [
        'retrieval_score', 'name_exact', 'name_levenshtein', 'name_jaro_winkler', 
        'name_jaccard', 'name_token_overlap', 'name_length_ratio', 'both_addr_present',
        'addr_exact', 'addr_levenshtein', 'addr_jaro_winkler', 'addr_jaccard', 
        'postal_match', 'house_num_match', 'name_addr_agree', 'name_postal_agree'
    ]
    
    # -------------------------------------------------------------
    # 3. Evaluate End-to-End Across K = [10, 20, 30, 40, 50]
    # -------------------------------------------------------------
    print("\n[4/4] Running 3-Fold GroupKFold Cross-Validation for Each K Budget...")
    
    k_comparison = []
    
    for k_val in [10, 20, 30, 40, 50]:
        print(f"\n" + "-" * 60)
        print(f"EVALUATING PIPELINE AT K = {k_val}")
        print("-" * 60)
        
        t_k0 = time.time()
        # Filter candidate pool to top k
        k_df = df_features[df_features['rank'] <= k_val].copy().reset_index(drop=True)
        
        # Candidate generation metrics
        hits = k_df['is_match'].sum()
        total_candidates = len(k_df)
        cand_recall = round(float(hits / total_eval_gt), 4)
        cand_precision = round(float(hits / total_candidates), 4)
        avg_cand = round(float(total_candidates / total_s1), 2)
        s1_counts = k_df.groupby('s1_id').size()
        p95_cand = float(s1_counts.quantile(0.95))
        p99_cand = float(s1_counts.quantile(0.99))
        
        print(f"  Candidate Recall:    {cand_recall*100:>6.2f}% ({hits:,}/{total_eval_gt:,} true hits)")
        print(f"  Candidate Precision: {cand_precision*100:>6.2f}% ({hits:,}/{total_candidates:,} candidates)")
        print(f"  Avg Candidates/S1:   {avg_cand} (P95: {p95_cand}, P99: {p99_cand})")
        
        # 3-Fold GroupKFold
        gkf = GroupKFold(n_splits=3)
        fold_scores = []
        oof_predictions = []
        oof_probs = np.zeros(len(k_df))
        
        t_train_total = 0.0
        t_infer_total = 0.0
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(k_df, groups=k_df['s1_id']), 1):
            X_tr, y_tr = k_df.iloc[train_idx][feature_cols], k_df.iloc[train_idx]['is_match']
            X_va, y_va = k_df.iloc[val_idx][feature_cols], k_df.iloc[val_idx]['is_match']
            
            val_s1_fold = list(k_df.iloc[val_idx]['s1_id'].unique())
            
            t0 = time.time()
            model = xgb.XGBClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold,
                eval_metric='logloss'
            )
            model.fit(X_tr, y_tr)
            t_train_total += time.time() - t0
            
            t0 = time.time()
            val_preds_prob = model.predict_proba(X_va)[:, 1]
            t_infer_total += time.time() - t0
            oof_probs[val_idx] = val_preds_prob
            
        k_df['oof_prob'] = oof_probs
        
        # Sweep threshold on Out-Of-Fold predictions
        best_th = 0.60
        best_macro_f05 = 0.0
        best_prec = 0.0
        best_rec = 0.0
        
        for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            pred_sub = k_df[k_df['oof_prob'] >= th][['s1_id', 'target_id']]
            m_f05 = calc_macro_f05(pred_sub, gt_dict, all_s1_eval)
            if m_f05 > best_macro_f05:
                best_macro_f05 = m_f05
                best_th = th
                p = precision_score(k_df['is_match'], (oof_probs >= th).astype(int), zero_division=0)
                r = recall_score(k_df['is_match'], (oof_probs >= th).astype(int), zero_division=0)
                best_prec = p
                best_rec = r
                
        oof_pr_auc = average_precision_score(k_df['is_match'], oof_probs)
        k_elapsed = time.time() - t_k0
        
        print(f"  >>> OUT-OF-FOLD RESULTS AT K = {k_val}:")
        print(f"      Best Threshold:  {best_th}")
        print(f"      Final Macro F0.5: {best_macro_f05:.4f}")
        print(f"      Pair Precision:  {best_prec:.4f}")
        print(f"      Pair Recall:     {best_rec:.4f}")
        print(f"      OOF PR-AUC:      {oof_pr_auc:.4f}")
        print(f"      Total Time:      {k_elapsed:.2f}s (Train: {t_train_total:.2f}s, Infer: {t_infer_total:.3f}s)")
        
        k_comparison.append({
            'K_budget': k_val,
            'candidate_recall': cand_recall,
            'candidate_precision': cand_precision,
            'total_candidates': total_candidates,
            'avg_candidates_per_s1': avg_cand,
            'p95_candidates': p95_cand,
            'optimal_threshold': best_th,
            'final_macro_f05': round(best_macro_f05, 4),
            'pair_precision': round(best_prec, 4),
            'pair_recall': round(best_rec, 4),
            'oof_pr_auc': round(oof_pr_auc, 4),
            'train_time_sec': round(t_train_total, 2),
            'infer_time_sec': round(t_infer_total, 3)
        })
        
    df_k_summary = pd.DataFrame(k_comparison)
    summary_path = os.path.join(REPORTS_DIR, "26_end_to_end_k_comparison.csv")
    df_k_summary.to_csv(summary_path, index=False)
    print(f"\n" + "=" * 75)
    print("FINAL END-TO-END K COMPARISON TABLE:")
    print("=" * 75)
    print(df_k_summary[['K_budget', 'candidate_recall', 'candidate_precision', 'avg_candidates_per_s1', 'optimal_threshold', 'final_macro_f05', 'pair_precision', 'pair_recall']])
    print(f"\nSaved report to: {summary_path}")
    print("\nEnd-to-End K-Sweep Complete!")

if __name__ == "__main__":
    run_step10()
