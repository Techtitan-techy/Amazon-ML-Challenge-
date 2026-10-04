import os
import sys
import time
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REPORTS_DIR

def calc_f_beta(precision, recall, beta=0.5):
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom

def calc_macro_f05(pred_df, gt_dict, all_s1_ids):
    """
    Computes true macro-averaged F0.5 per S1 entity as specified by the Amazon ML Challenge:
    - Singletons with 0 true matches and 0 predicted matches get 1.0
    - Singletons with false matches get 0.0
    - Average across ALL S1 entities in the evaluation set
    """
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

def run_step05():
    print("=" * 60)
    print("Running Step 5: Model Benchmark, Thresholds, Margins & Singletons")
    print("=" * 60)
    
    true_parquet = os.path.join(REPORTS_DIR, "08_true_pair_features.parquet")
    hard_parquet = os.path.join(REPORTS_DIR, "09_hard_negative_analysis.parquet")
    
    if not os.path.exists(true_parquet) or not os.path.exists(hard_parquet):
        print("Feature parquet files not found! Run Step 4 first.")
        return
        
    print("[1/4] Loading features for modeling...")
    df_true = pd.read_parquet(true_parquet)
    df_hard = pd.read_parquet(hard_parquet)
    
    df_all = pd.concat([df_true, df_hard], ignore_index=True)
    
    feature_cols = [
        'name_exact', 'name_levenshtein', 'name_jaro_winkler', 'name_jaccard', 
        'name_token_overlap', 'name_length_ratio', 'both_addr_present',
        'addr_exact', 'addr_levenshtein', 'addr_jaro_winkler', 'addr_jaccard', 
        'postal_match', 'house_num_match', 'name_addr_agree', 'name_postal_agree'
    ]
    
    # -------------------------------------------------------------
    # 1. Group-aware Train/Validation Split (Phase 12)
    # -------------------------------------------------------------
    print("[2/4] Splitting by S1 entity group to prevent leakage...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(gss.split(df_all, groups=df_all['s1_id']))
    
    train_df = df_all.iloc[train_idx].copy()
    val_df = df_all.iloc[val_idx].copy()
    
    X_train, y_train = train_df[feature_cols], train_df['is_match']
    X_val, y_val = val_df[feature_cols], val_df['is_match']
    
    val_s1_all = list(val_df['s1_id'].unique())
    val_gt_dict = val_df[val_df['is_match'] == 1].groupby('s1_id')['target_id'].apply(set).to_dict()
    
    # Benchmark models: Logistic Regression, LightGBM, XGBoost
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=6, random_state=42, eval_metric='logloss')
    }
    
    bench_results = []
    trained_models = {}
    
    for m_name, model in models.items():
        t0 = time.time()
        print(f"  Training {m_name}...")
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t_infer = time.time()
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
        else:
            probs = model.decision_function(X_val)
        infer_time = time.time() - t_infer
        
        trained_models[m_name] = (model, probs)
        
        # Binary pair metrics at default 0.5
        preds_binary = (probs >= 0.5).astype(int)
        pair_prec = precision_score(y_val, preds_binary, zero_division=0)
        pair_rec = recall_score(y_val, preds_binary, zero_division=0)
        pair_f05 = calc_f_beta(pair_prec, pair_rec, beta=0.5)
        pr_auc = average_precision_score(y_val, probs)
        
        # Macro F0.5 per S1
        val_pred_df = val_df[probs >= 0.5][['s1_id', 'target_id']]
        macro_f05 = calc_macro_f05(val_pred_df, val_gt_dict, val_s1_all)
        
        bench_results.append({
            'model': m_name,
            'precision': round(float(pair_prec), 4),
            'recall': round(float(pair_rec), 4),
            'pair_f05': round(float(pair_f05), 4),
            'macro_f05': round(float(macro_f05), 4),
            'pr_auc': round(float(pr_auc), 4),
            'train_time_sec': round(float(train_time), 2),
            'infer_time_sec': round(float(infer_time), 3),
        })
        print(f"    Macro F0.5: {macro_f05:.4f} | PR-AUC: {pr_auc:.4f} | Train: {train_time:.2f}s")
        
    df_bench = pd.DataFrame(bench_results)
    bench_out = os.path.join(REPORTS_DIR, "14_model_benchmark.csv")
    df_bench.to_csv(bench_out, index=False)
    print(f"  --> Saved {bench_out}")
    print(df_bench)

    # -------------------------------------------------------------
    # 2. Threshold Search (Phase 13)
    # -------------------------------------------------------------
    print("\n[3/4] Threshold search on best model (LightGBM)...")
    best_probs = trained_models['LightGBM'][1]
    val_df['prob'] = best_probs
    
    thresholds = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]
    thresh_rows = []
    
    for thresh in thresholds:
        preds_t = (best_probs >= thresh).astype(int)
        p = precision_score(y_val, preds_t, zero_division=0)
        r = recall_score(y_val, preds_t, zero_division=0)
        f05 = calc_f_beta(p, r, beta=0.5)
        
        pred_sub = val_df[best_probs >= thresh][['s1_id', 'target_id']]
        m_f05 = calc_macro_f05(pred_sub, val_gt_dict, val_s1_all)
        
        thresh_rows.append({
            'threshold': thresh,
            'precision': round(float(p), 4),
            'recall': round(float(r), 4),
            'pair_f05': round(float(f05), 4),
            'macro_f05': round(float(m_f05), 4)
        })
        
    df_thresh = pd.DataFrame(thresh_rows)
    thresh_out = os.path.join(REPORTS_DIR, "15_threshold_search.csv")
    df_thresh.to_csv(thresh_out, index=False)
    print(f"  --> Saved {thresh_out}")
    print(df_thresh.sort_values(by='macro_f05', ascending=False).head(5))

    # -------------------------------------------------------------
    # 3. Margin & Singleton Analysis (Phases 13 & 14)
    # -------------------------------------------------------------
    print("\n[4/4] Computing Score Margin and Singleton Behavior...")
    
    # For each S1 in val, compute best score, second best score, margin
    margin_rows = []
    singleton_rows = []
    
    for s1, group in val_df.groupby('s1_id'):
        sorted_probs = group['prob'].sort_values(ascending=False).values
        best_score = float(sorted_probs[0])
        second_score = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = round(best_score - second_score, 4)
        
        true_matches = group['is_match'].sum()
        cardinality_bucket = '0_matches' if true_matches == 0 else ('1_match' if true_matches == 1 else '2+_matches')
        
        margin_rows.append({
            's1_id': s1,
            'best_score': round(best_score, 4),
            'second_score': round(second_score, 4),
            'margin': margin,
            'true_matches': int(true_matches),
            'cardinality_bucket': cardinality_bucket
        })
        
    df_margin = pd.DataFrame(margin_rows)
    
    # Margin quantiles & safety
    margin_summary = df_margin.groupby('cardinality_bucket').agg(
        avg_best_score=('best_score', 'mean'),
        median_best_score=('best_score', 'median'),
        avg_second_score=('second_score', 'mean'),
        avg_margin=('margin', 'mean'),
        median_margin=('margin', 'median'),
        count=('s1_id', 'count')
    ).reset_index()
    
    margin_out = os.path.join(REPORTS_DIR, "16_margin_analysis.csv")
    margin_summary.to_csv(margin_out, index=False)
    print(f"  --> Saved {margin_out}")
    print(margin_summary)
    
    # Singleton Gate analysis: can we reliably identify singletons by best_score & margin?
    singleton_gate = df_margin.copy()
    singleton_gate['predicted_singleton'] = (singleton_gate['best_score'] < 0.4) | (
        (singleton_gate['best_score'] < 0.6) & (singleton_gate['margin'] < 0.1)
    )
    singleton_acc = (singleton_gate['predicted_singleton'] == (singleton_gate['true_matches'] == 0)).mean()
    
    singleton_report = df_margin.groupby('cardinality_bucket').agg(
        s1_count=('s1_id', 'count'),
        pct_best_score_above_80=('best_score', lambda x: round(float((x >= 0.8).mean() * 100), 2)),
        pct_best_score_below_40=('best_score', lambda x: round(float((x < 0.4).mean() * 100), 2)),
        mean_margin=('margin', lambda x: round(float(x.mean()), 4))
    ).reset_index()
    
    singleton_out = os.path.join(REPORTS_DIR, "17_singleton_analysis.csv")
    singleton_report.to_csv(singleton_out, index=False)
    print(f"  --> Saved {singleton_out}")
    print(singleton_report)
    print("\nStep 5 Complete!")

if __name__ == "__main__":
    run_step05()
