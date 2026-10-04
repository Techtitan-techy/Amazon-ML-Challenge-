import os
import sys
import re
import time
import json
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, REPORTS_DIR
from normalization import transliterated, extract_postal, extract_house_num

# Indian state acronym expansions (bidirectional / general)
STATE_ABBR_MAP = {
    r'\bwb\b': 'west bengal',
    r'\bmh\b': 'maharashtra',
    r'\bmp\b': 'madhya pradesh',
    r'\bup\b': 'uttar pradesh',
    r'\bdl\b': 'delhi',
    r'\bka\b': 'karnataka',
    r'\btn\b': 'tamil nadu',
    r'\bts\b': 'telangana',
    r'\bap\b': 'andhra pradesh',
    r'\bgj\b': 'gujarat',
    r'\brj\b': 'rajasthan',
    r'\bpb\b': 'punjab',
    r'\bhr\b': 'haryana',
    r'\bkl\b': 'kerala',
}

def expand_state_abbreviations(addr: str) -> str:
    if not addr:
        return ""
    s = addr.lower()
    for pat, exp in STATE_ABBR_MAP.items():
        s = re.sub(pat, exp, s)
    return s

def indic_phonetic_clean(text: str) -> str:
    if not text:
        return ""
    s = text.lower()
    s = re.sub(r'ph', 'f', s)
    s = re.sub(r'ee|ea|ey', 'i', s)
    s = re.sub(r'oo|ou', 'u', s)
    s = re.sub(r'kh|gh|dh|th|bh|jh|sh', lambda m: m.group(0)[0], s)
    s = re.sub(r'w', 'v', s)
    s = re.sub(r'z', 's', s)
    s = re.sub(r'(.)\1+', r'\1', s)
    return s

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

def batch_top_k(query_matrix, index_matrix, k=20, batch_size=500):
    n_queries = query_matrix.shape[0]
    results = []
    
    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        batch_queries = query_matrix[start_idx:end_idx]
        sim_matrix = batch_queries.dot(index_matrix.T).tocsr()
        
        for q_offset in range(end_idx - start_idx):
            q_idx = start_idx + q_offset
            r_start = sim_matrix.indptr[q_offset]
            r_end = sim_matrix.indptr[q_offset + 1]
            if r_start == r_end:
                continue
            
            target_indices = sim_matrix.indices[r_start:r_end]
            scores = sim_matrix.data[r_start:r_end]
            
            if len(scores) > k:
                top_k_part = np.argpartition(-scores, k)[:k]
                top_target_idx = target_indices[top_k_part]
                top_scores = scores[top_k_part]
                sort_order = np.argsort(-top_scores)
                top_target_idx = top_target_idx[sort_order]
                top_scores = top_scores[sort_order]
            else:
                sort_order = np.argsort(-scores)
                top_target_idx = target_indices[sort_order]
                top_scores = scores[sort_order]
                
            for t_idx, s in zip(top_target_idx, top_scores):
                if s > 0.05:
                    results.append((q_idx, t_idx, float(s)))
                    
    return results

def run_step11():
    print("=" * 75)
    print("STEP 11: SPECIALIST RESIDUAL RETRIEVAL BENCHMARK")
    print("Testing Phonetic, Missing-Address Fallback & Structured State Expansion")
    print("=" * 75)
    
    # 1. Load exact GT dict
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
    print(f"Loaded {total_s1:,} S1 evaluation entities with {total_eval_gt:,} true Ground Truth pairs.")
    
    # 2. Load the baseline candidate dataset (already computed in step 7)
    cand_path = os.path.join(REPORTS_DIR, "21_improved_candidates.parquet")
    df_baseline_cand = pd.read_parquet(cand_path)
    print(f"Baseline Candidate Pool: {len(df_baseline_cand):,} pairs.")
    
    # Connect DuckDB to load the texts for S1 and targets in cohort
    con = duckdb.connect()
    con.register('df_cand_s1', pd.DataFrame({'s1_id': all_s1_eval}))
    
    eval_s1_df = con.execute(f"""
        SELECT s.entity_id as s1_id, s.business_name as s1_name, s.business_address as s1_addr, s.country
        FROM read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True) s
        JOIN df_cand_s1 c ON s.entity_id = c.s1_id
    """).fetch_df()
    
    # Load targets that appeared in candidates or true targets
    all_target_ids = list(set(df_baseline_cand['target_id']).union({tgt for tgts in gt_dict.values() for tgt in tgts}))
    con.register('df_target_ids', pd.DataFrame({'entity_id': all_target_ids}))
    
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE targets_pool AS
        SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True)
        UNION ALL
        SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True)
    """)
    
    targets_df = con.execute("""
        SELECT t.*
        FROM targets_pool t
        JOIN df_target_ids i ON t.entity_id = i.entity_id
    """).fetch_df()
    print(f"Loaded {len(targets_df):,} target records for specialist indexing.")
    
    # Pre-compute normalizations
    print("\nPreparing specialist normalizations...")
    eval_s1_df['norm_name'] = eval_s1_df['s1_name'].apply(transliterated)
    eval_s1_df['norm_addr'] = eval_s1_df['s1_addr'].apply(transliterated)
    eval_s1_df['addr_expanded'] = eval_s1_df['norm_addr'].apply(expand_state_abbreviations)
    eval_s1_df['phonetic_name'] = eval_s1_df['norm_name'].apply(indic_phonetic_clean)
    
    targets_df['norm_name'] = targets_df['business_name'].apply(transliterated)
    targets_df['norm_addr'] = targets_df['business_address'].apply(transliterated)
    targets_df['addr_expanded'] = targets_df['norm_addr'].apply(expand_state_abbreviations)
    targets_df['phonetic_name'] = targets_df['norm_name'].apply(indic_phonetic_clean)
    targets_df['is_addr_missing'] = targets_df['norm_addr'].str.strip() == ""
    
    # -------------------------------------------------------------
    # Build Specialist Retrieval Channels by Country
    # -------------------------------------------------------------
    phonetic_pairs = set()
    missing_addr_pairs = set()
    expanded_addr_pairs = set()
    
    for country in ['India', 'US']:
        print(f"\n--- Indexing Specialist Channels for: {country} ---")
        q_sub = eval_s1_df[eval_s1_df['country'] == country].reset_index(drop=True)
        t_sub = targets_df[targets_df['country'] == country].reset_index(drop=True)
        q_ids = q_sub['s1_id'].values
        t_ids = t_sub['entity_id'].values
        
        # 1. Indic Phonetic Name Retrieval (Char 3-4 grams on phonetic_name)
        if country == 'India':
            print("  Building Channel A: Indic Phonetic Name TF-IDF...")
            ph_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=50000)
            t_ph_mat = ph_vec.fit_transform(t_sub['phonetic_name'])
            q_ph_mat = ph_vec.transform(q_sub['phonetic_name'])
            ph_res = batch_top_k(q_ph_mat, t_ph_mat, k=10, batch_size=1000)
            for q_i, t_i, s in ph_res:
                if s >= 0.25:
                    phonetic_pairs.add((q_ids[q_i], t_ids[t_i]))
            print(f"    Indic Phonetic candidates: {len(phonetic_pairs):,} pairs.")
            
        # 2. Missing-Address Fallback Channel (Search S1 Name against targets where address is empty)
        print("  Building Channel B: Missing-Address Fallback Channel...")
        missing_mask = t_sub['is_addr_missing'].values
        if missing_mask.sum() > 0:
            t_miss_sub = t_sub[missing_mask].reset_index(drop=True)
            t_miss_ids = t_miss_sub['entity_id'].values
            
            miss_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=40000)
            t_miss_mat = miss_vec.fit_transform(t_miss_sub['norm_name'])
            q_miss_mat = miss_vec.transform(q_sub['norm_name'])
            miss_res = batch_top_k(q_miss_mat, t_miss_mat, k=5, batch_size=1000)
            for q_i, t_i, s in miss_res:
                if s >= 0.40: # High name precision required when address missing
                    missing_addr_pairs.add((q_ids[q_i], t_miss_ids[t_i]))
            print(f"    Missing-Address Fallback candidates: {len(missing_addr_pairs):,} pairs.")
            
        # 3. Structured State Expansion Channel (Word TF-IDF on expanded address)
        print("  Building Channel C: Structured State Expansion Channel...")
        state_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\b\w+\b', min_df=2, max_features=60000)
        t_state_mat = state_vec.fit_transform(t_sub['norm_name'] + " " + t_sub['addr_expanded'])
        q_state_mat = state_vec.transform(q_sub['norm_name'] + " " + q_sub['addr_expanded'])
        state_res = batch_top_k(q_state_mat, t_state_mat, k=10, batch_size=1000)
        for q_i, t_i, s in state_res:
            if s >= 0.30:
                expanded_addr_pairs.add((q_ids[q_i], t_ids[t_i]))
        print(f"    Expanded Address candidates: {len(expanded_addr_pairs):,} pairs.")
        
    # Check how many of the 477 missed pairs are recovered by each channel!
    missed_csv = os.path.join(REPORTS_DIR, "24_missed_gt_deepdive.csv")
    df_missed = pd.read_csv(missed_csv)
    missed_set = set(zip(df_missed['s1_id'], df_missed['target_id']))
    
    hits_ph = len(phonetic_pairs.intersection(missed_set))
    hits_miss = len(missing_addr_pairs.intersection(missed_set))
    hits_exp = len(expanded_addr_pairs.intersection(missed_set))
    hits_all = len((phonetic_pairs | missing_addr_pairs | expanded_addr_pairs).intersection(missed_set))
    
    print("\n" + "=" * 60)
    print("RECOVERY OF THE 477 PREVIOUSLY MISSED PAIRS:")
    print("=" * 60)
    print(f"  Channel A (Phonetic Name):        +{hits_ph} pairs recovered ({hits_ph/len(missed_set)*100:.2f}%)")
    print(f"  Channel B (Missing-Address):       +{hits_miss} pairs recovered ({hits_miss/len(missed_set)*100:.2f}%)")
    print(f"  Channel C (Expanded State):       +{hits_exp} pairs recovered ({hits_exp/len(missed_set)*100:.2f}%)")
    print(f"  Channels A + B + C Combined:       +{hits_all} pairs recovered ({hits_all/len(missed_set)*100:.2f}%)")
    
    # -------------------------------------------------------------
    # End-to-End Evaluation on Final Macro F0.5
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("END-TO-END MACRO F0.5 BENCHMARK ACROSS CONFIGURATIONS")
    print("=" * 60)
    
    # Extract raw text dictionaries for fast feature lookup
    s1_text_map = eval_s1_df.set_index('s1_id')[['norm_name', 'norm_addr', 'country']].to_dict('index')
    tgt_text_map = targets_df.set_index('entity_id')[['norm_name', 'norm_addr', 'country']].to_dict('index')
    
    baseline_pairs_k30 = set(zip(df_baseline_cand[df_baseline_cand['rank'] <= 30]['s1_id'], df_baseline_cand[df_baseline_cand['rank'] <= 30]['target_id']))
    baseline_pairs_k50 = set(zip(df_baseline_cand[df_baseline_cand['rank'] <= 50]['s1_id'], df_baseline_cand[df_baseline_cand['rank'] <= 50]['target_id']))
    
    configs = [
        ("Baseline_K30", baseline_pairs_k30),
        ("K30 + Indic_Phonetic", baseline_pairs_k30 | phonetic_pairs),
        ("K30 + Missing_Addr_Fallback", baseline_pairs_k30 | missing_addr_pairs),
        ("K30 + State_Expansion", baseline_pairs_k30 | expanded_addr_pairs),
        ("K30 + All_Specialist", baseline_pairs_k30 | phonetic_pairs | missing_addr_pairs | expanded_addr_pairs),
        ("Baseline_K50", baseline_pairs_k50),
        ("K50 + All_Specialist", baseline_pairs_k50 | phonetic_pairs | missing_addr_pairs | expanded_addr_pairs),
    ]
    
    feature_cols = [
        'name_exact', 'name_levenshtein', 'name_jaro_winkler', 
        'name_jaccard', 'name_token_overlap', 'name_length_ratio', 'both_addr_present',
        'addr_exact', 'addr_levenshtein', 'addr_jaro_winkler', 'addr_jaccard', 
        'postal_match', 'house_num_match', 'name_addr_agree', 'name_postal_agree'
    ]
    
    config_results = []
    
    for cfg_name, cand_pair_set in configs:
        t0 = time.time()
        n_cands = len(cand_pair_set)
        hits = len(cand_pair_set.intersection(gt_pair_set))
        recall = round(float(hits / total_eval_gt), 4)
        prec = round(float(hits / n_cands), 4)
        avg_cand = round(float(n_cands / total_s1), 2)
        
        # Build features for this candidate set
        df_cfg = pd.DataFrame(list(cand_pair_set), columns=['s1_id', 'target_id'])
        
        # Compute features
        feats = []
        for _, r in df_cfg.iterrows():
            s1_id, tgt_id = r['s1_id'], r['target_id']
            s1_info = s1_text_map.get(s1_id, {'norm_name': '', 'norm_addr': '', 'country': ''})
            tgt_info = tgt_text_map.get(tgt_id, {'norm_name': '', 'norm_addr': '', 'country': ''})
            
            f = compute_pair_features(
                s1_info['norm_name'], tgt_info['norm_name'],
                s1_info['norm_addr'], tgt_info['norm_addr'],
                s1_info['country'], tgt_info['country'],
                0.5 # default score
            )
            f['s1_id'] = s1_id
            f['target_id'] = tgt_id
            f['is_match'] = 1 if (s1_id, tgt_id) in gt_pair_set else 0
            feats.append(f)
            
        df_feat = pd.DataFrame(feats)
        
        # 3-Fold GroupKFold
        gkf = GroupKFold(n_splits=3)
        oof_probs = np.zeros(len(df_feat))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_feat, groups=df_feat['s1_id'])):
            X_tr, y_tr = df_feat.iloc[train_idx][feature_cols], df_feat.iloc[train_idx]['is_match']
            X_va, y_va = df_feat.iloc[val_idx][feature_cols], df_feat.iloc[val_idx]['is_match']
            
            model = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.08, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42 + fold,
                eval_metric='logloss'
            )
            model.fit(X_tr, y_tr)
            oof_probs[val_idx] = model.predict_proba(X_va)[:, 1]
            
        df_feat['oof_prob'] = oof_probs
        
        # Decision at optimal threshold 0.70
        pred_sub = df_feat[df_feat['oof_prob'] >= 0.70][['s1_id', 'target_id']]
        m_f05 = calc_macro_f05(pred_sub, gt_dict, all_s1_eval)
        p_eval = precision_score(df_feat['is_match'], (oof_probs >= 0.70).astype(int), zero_division=0)
        r_eval = recall_score(df_feat['is_match'], (oof_probs >= 0.70).astype(int), zero_division=0)
        
        elapsed = time.time() - t0
        print(f"  {cfg_name:28s} | Recall: {recall*100:>6.2f}% | Cand/S1: {avg_cand:>5.2f} | Macro F0.5: {m_f05:.4f} | Prec: {p_eval:.4f} | Rec: {r_eval:.4f} ({elapsed:.1f}s)")
        
        config_results.append({
            'configuration': cfg_name,
            'candidate_recall': recall,
            'candidate_precision': prec,
            'avg_candidates_per_s1': avg_cand,
            'total_candidates': n_cands,
            'final_macro_f05': round(m_f05, 4),
            'pair_precision': round(p_eval, 4),
            'pair_recall': round(r_eval, 4),
            'elapsed_sec': round(elapsed, 1)
        })
        
    df_results = pd.DataFrame(config_results)
    out_csv = os.path.join(REPORTS_DIR, "27_specialist_channels_benchmark.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n  --> Saved {out_csv}")
    print("\n" + "=" * 75)
    print("SPECIALIST BENCHMARK SUMMARY TABLE:")
    print("=" * 75)
    print(df_results[['configuration', 'candidate_recall', 'avg_candidates_per_s1', 'final_macro_f05', 'pair_precision', 'pair_recall']])
    print("\nStep 11 Complete!")

if __name__ == "__main__":
    run_step11()
