import os
import sys
import time
import json
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, REPORTS_DIR
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

def compute_pair_features(s1_name, tgt_name, s1_addr, tgt_addr, s1_country, tgt_country, retrieval_score):
    s1_n = transliterated(s1_name)
    tgt_n = transliterated(tgt_name)
    s1_a = transliterated(s1_addr)
    tgt_a = transliterated(tgt_addr)
    
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
    
    p1 = extract_postal(s1_addr, s1_country)
    p2 = extract_postal(tgt_addr, tgt_country)
    postal_match = 1.0 if (p1 and p2 and p1 == p2) else 0.0
    
    h1 = extract_house_num(s1_addr)
    h2 = extract_house_num(tgt_addr)
    house_match = 1.0 if (h1 and h2 and h1 == h2) else 0.0
    
    return {
        'retrieval_score': round(float(retrieval_score), 4),
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

def run_step08():
    print("=" * 70)
    print("STEPS 5 - 7: MODELING ON FUZZY CANDIDATES, THRESHOLDS & RESIDUAL GAP")
    print("=" * 70)
    
    cand_path = os.path.join(REPORTS_DIR, "21_improved_candidates.parquet")
    if not os.path.exists(cand_path):
        print(f"Error: {cand_path} does not exist. Run step07 first.")
        return
        
    con = duckdb.connect()
    
    print("\n[1/4] Loading improved candidate distribution and joining raw texts...")
    t0 = time.time()
    
    cand_df = pd.read_parquet(cand_path)
    con.register('cand_df', cand_df)
    
    joined_df = con.execute(f"""
        WITH targets_all AS (
            SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True)
            UNION ALL
            SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True)
        )
        SELECT 
            c.s1_id,
            c.target_id,
            c.score as retrieval_score,
            c.is_match,
            s1.business_name as s1_name,
            s1.business_address as s1_addr,
            s1.country as s1_country,
            t.business_name as tgt_name,
            t.business_address as tgt_addr,
            t.country as tgt_country
        FROM cand_df c
        JOIN read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True) s1 ON c.s1_id = s1.entity_id
        JOIN targets_all t ON c.target_id = t.entity_id
    """).fetch_df()
    
    print(f"  Loaded {len(joined_df):,} candidate pairs in {time.time()-t0:.2f}s.")
    print(f"  Positives: {joined_df['is_match'].sum():,} | Negatives: {(joined_df['is_match']==0).sum():,}")
    
    print("\n[2/4] Computing feature matrix...")
    t0 = time.time()
    feature_rows = []
    for _, r in joined_df.iterrows():
        feats = compute_pair_features(
            r['s1_name'], r['tgt_name'],
            r['s1_addr'], r['tgt_addr'],
            r['s1_country'], r['tgt_country'],
            r['retrieval_score']
        )
        feats['s1_id'] = r['s1_id']
        feats['target_id'] = r['target_id']
        feats['is_match'] = r['is_match']
        feature_rows.append(feats)
        
    df_features = pd.DataFrame(feature_rows)
    print(f"  Feature generation completed in {time.time()-t0:.2f}s.")
    
    # -------------------------------------------------------------
    # 5. Train XGBoost on Improved Candidate Distribution
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: XGBOOST ON IMPROVED FUZZY CANDIDATE DISTRIBUTION")
    print("=" * 60)
    
    feature_cols = [
        'retrieval_score', 'name_exact', 'name_levenshtein', 'name_jaro_winkler', 
        'name_jaccard', 'name_token_overlap', 'name_length_ratio', 'both_addr_present',
        'addr_exact', 'addr_levenshtein', 'addr_jaro_winkler', 'addr_jaccard', 
        'postal_match', 'house_num_match', 'name_addr_agree', 'name_postal_agree'
    ]
    
    # Group split on s1_id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(gss.split(df_features, groups=df_features['s1_id']))
    
    train_data = df_features.iloc[train_idx].copy()
    val_data = df_features.iloc[val_idx].copy()
    
    X_train, y_train = train_data[feature_cols], train_data['is_match']
    X_val, y_val = val_data[feature_cols], val_data['is_match']
    
    val_s1_all = list(val_data['s1_id'].unique())
    val_gt_dict = val_data[val_data['is_match'] == 1].groupby('s1_id')['target_id'].apply(set).to_dict()
    
    print(f"  Train: {len(X_train):,} pairs ({y_train.sum():,} pos) | Val: {len(X_val):,} pairs ({y_val.sum():,} pos)")
    
    # Train XGBoost
    print("  Fitting XGBoost model...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=150, 
        learning_rate=0.08, 
        max_depth=6, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Predict probabilities
    t0 = time.time()
    val_probs = model.predict_proba(X_val)[:, 1]
    infer_time = time.time() - t0
    val_data['prob'] = val_probs
    
    pr_auc = average_precision_score(y_val, val_probs)
    print(f"  XGBoost trained in {train_time:.2f}s | Val PR-AUC: {pr_auc:.4f}")
    
    # Feature importances
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    print("\nTop 5 Feature Importances:")
    print(imp_df.head(5))
    
    # -------------------------------------------------------------
    # 6. Entity-Level Decision & Threshold Experiment
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: ENTITY-LEVEL THRESHOLD & DECISION POLICY EXPERIMENT")
    print("=" * 60)
    
    thresholds = [round(x, 2) for x in np.arange(0.40, 0.96, 0.05)]
    results_policy = []
    
    for th in thresholds:
        preds = val_data[val_data['prob'] >= th][['s1_id', 'target_id']]
        m_f05 = calc_macro_f05(preds, val_gt_dict, val_s1_all)
        p = precision_score(y_val, (val_probs >= th).astype(int), zero_division=0)
        r = recall_score(y_val, (val_probs >= th).astype(int), zero_division=0)
        results_policy.append({
            'threshold': th,
            'macro_f05': round(m_f05, 4),
            'pair_precision': round(p, 4),
            'pair_recall': round(r, 4),
            'pair_f05': round(calc_f_beta(p, r, beta=0.5), 4)
        })
        
    df_policy = pd.DataFrame(results_policy)
    best_row = df_policy.sort_values(by='macro_f05', ascending=False).iloc[0]
    print("\nThreshold Sweep (Top 5):")
    print(df_policy.sort_values(by='macro_f05', ascending=False).head(5))
    print(f"\n>>> Optimal Flat Threshold: {best_row['threshold']} -> Macro F0.5: {best_row['macro_f05']:.4f} <<<")
    
    # Now evaluate Score + Margin Rule
    print("\n--- Evaluating Score + Margin + Singleton Gate Policy ---")
    margin_gate_results = []
    for base_th in [0.70, 0.75, 0.80, 0.85]:
        for min_margin in [0.0, 0.05, 0.10, 0.15]:
            # Apply policy per entity:
            # 1. Filter candidates >= base_th
            # 2. If entity has 1 candidate and best_score < 0.85, require margin >= min_margin
            accepted_pairs = []
            for s1, group in val_data.groupby('s1_id'):
                g_sorted = group.sort_values(by='prob', ascending=False)
                best_prob = g_sorted.iloc[0]['prob']
                
                # Singleton floor gate:
                if best_prob < 0.40:
                    continue # Predict empty
                    
                second_prob = g_sorted.iloc[1]['prob'] if len(g_sorted) > 1 else 0.0
                margin = best_prob - second_prob
                
                # Match selection
                for _, row in g_sorted.iterrows():
                    p_val = row['prob']
                    if p_val >= base_th:
                        # If border zone, check margin
                        if p_val < 0.85 and margin < min_margin:
                            continue
                        accepted_pairs.append((row['s1_id'], row['target_id']))
                        
            df_acc = pd.DataFrame(accepted_pairs, columns=['s1_id', 'target_id'])
            m_f05_gate = calc_macro_f05(df_acc, val_gt_dict, val_s1_all)
            margin_gate_results.append({
                'base_threshold': base_th,
                'min_margin': min_margin,
                'macro_f05': round(m_f05_gate, 4)
            })
            
    df_margin_gate = pd.DataFrame(margin_gate_results).sort_values(by='macro_f05', ascending=False)
    print("\nTop 5 Margin Gate Configurations:")
    print(df_margin_gate.head(5))
    
    # Save decision policies report
    policy_out_path = os.path.join(REPORTS_DIR, "22_decision_policy_experiment.csv")
    df_margin_gate.to_csv(policy_out_path, index=False)
    print(f"  --> Saved {policy_out_path}")
    
    # -------------------------------------------------------------
    # 7. Residual Recall Gap & Embedding / ANN Analysis
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: RESIDUAL RECALL GAP & EMBEDDINGS/ANN ASSESSMENT")
    print("=" * 60)
    
    missed_report_p = os.path.join(REPORTS_DIR, "19_missed_gt_analysis.csv")
    hybrid_report_p = os.path.join(REPORTS_DIR, "18_fuzzy_hybrid_recall.csv")
    
    df_hyb = pd.read_csv(hybrid_report_p)
    df_miss = pd.read_csv(missed_report_p)
    
    final_hybrid_recall = float(df_hyb[df_hyb['channel'] == 'HYBRID_UNION_ALL']['recall'].values[0])
    residual_gap_pct = round((1.0 - final_hybrid_recall) * 100, 2)
    
    print(f"\nFinal Hybrid Retrieval Recall: {final_hybrid_recall*100:.2f}%")
    print(f"Residual Recall Gap:           {residual_gap_pct:.2f}%")
    
    # Assess missed cases
    missing_addr_cnt = df_miss['tgt_addr_missing'].sum() if 'tgt_addr_missing' in df_miss.columns else 0
    missing_addr_pct = round(missing_addr_cnt / len(df_miss) * 100, 1) if len(df_miss) > 0 else 0.0
    
    len_diff_cnt = df_miss['extreme_len_diff'].sum() if 'extreme_len_diff' in df_miss.columns else 0
    len_diff_pct = round(len_diff_cnt / len(df_miss) * 100, 1) if len(df_miss) > 0 else 0.0
    
    assessment = []
    assessment.append(f"- **Classical Fuzzy Hybrid Retrieval achieves {final_hybrid_recall*100:.2f}% Candidate Recall** on the evaluation cohort.")
    assessment.append(f"- The total residual recall gap across the entire dataset is only **{residual_gap_pct:.2f}%**.")
    assessment.append(f"- Of the remaining unretrieved pairs, **{missing_addr_pct}% have completely missing addresses** in S2/S3, and **{len_diff_pct}% have extreme length differences** (abbreviations / acronyms).")
    
    if residual_gap_pct < 4.0:
        recommendation = (
            f"**Conclusion**: The residual recall gap is small ({residual_gap_pct}%). "
            f"Classical character n-gram + token TF-IDF + exact union captures over {final_hybrid_recall*100:.1f}% of true matches at zero GPU cost, "
            f"sub-second latency, and 100% deterministic reproducibility for the competition code audit. "
            f"Heavy embedding models / ANN are NOT strictly required to achieve top leaderboard performance, "
            f"and would introduce substantial inference latency and out-of-memory risks on 10.32M records without closing the missing-address gap."
        )
    else:
        recommendation = (
            f"**Conclusion**: The residual recall gap is {residual_gap_pct}%. "
            f"A lightweight bi-encoder (e.g. MiniLM-L6) targeted strictly at the missing-address or extreme acronym subset can be tested as an auxiliary channel."
        )
        
    print(recommendation)
    
    gap_summary = {
        'hybrid_candidate_recall': final_hybrid_recall,
        'residual_gap_pct': residual_gap_pct,
        'pct_missed_missing_address': missing_addr_pct,
        'pct_missed_acronyms_length': len_diff_pct,
        'ann_required': bool(residual_gap_pct >= 4.0),
        'recommendation': recommendation
    }
    gap_out_path = os.path.join(REPORTS_DIR, "23_residual_gap_and_ann_assessment.json")
    with open(gap_out_path, "w", encoding="utf-8") as f:
        json.dump(gap_summary, f, indent=2)
    print(f"  --> Saved {gap_out_path}")
    
    print("\nSteps 5 - 7 Complete!")

if __name__ == "__main__":
    run_step08()
