import os
import sys
import time
import duckdb
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, REPORTS_DIR
from normalization import raw, light_normalized, transliterated, extract_postal, extract_house_num

def compute_string_features(s1_name, tgt_name, s1_addr, tgt_addr, s1_country, tgt_country):
    s1_n = transliterated(s1_name)
    tgt_n = transliterated(tgt_name)
    s1_a = transliterated(s1_addr)
    tgt_a = transliterated(tgt_addr)
    
    # Name metrics
    n_exact = 1.0 if (s1_n and s1_n == tgt_n) else 0.0
    n_lev = Levenshtein.normalized_similarity(s1_n, tgt_n) if (s1_n and tgt_n) else 0.0
    n_jw = JaroWinkler.similarity(s1_n, tgt_n) if (s1_n and tgt_n) else 0.0
    
    s1_n_toks = set(s1_n.split())
    tgt_n_toks = set(tgt_n.split())
    n_jaccard = len(s1_n_toks & tgt_n_toks) / len(s1_n_toks | tgt_n_toks) if (s1_n_toks and tgt_n_toks) else 0.0
    n_overlap = len(s1_n_toks & tgt_n_toks) / min(len(s1_n_toks), len(tgt_n_toks)) if (s1_n_toks and tgt_n_toks) else 0.0
    n_len_ratio = min(len(s1_n), len(tgt_n)) / max(len(s1_n), len(tgt_n)) if max(len(s1_n), len(tgt_n)) > 0 else 0.0
    
    # Address metrics
    has_addr_s1 = 1.0 if s1_a else 0.0
    has_addr_tgt = 1.0 if tgt_a else 0.0
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
    
    # Cross-field
    name_addr_agree = 1.0 if (n_exact == 1.0 and a_exact == 1.0) else 0.0
    name_postal_agree = 1.0 if (n_exact == 1.0 and postal_match == 1.0) else 0.0
    
    return {
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
        'name_addr_agree': name_addr_agree,
        'name_postal_agree': name_postal_agree,
    }

def run_step04():
    print("=" * 60)
    print("Running Step 4: True Pairs, Hard Negatives & Feature Separability")
    print("=" * 60)
    
    con = duckdb.connect()
    
    # -------------------------------------------------------------
    # 1. Ground Truth Pair Analysis (Phase 6)
    # -------------------------------------------------------------
    print("\n[1/6] Extracting and profiling True Matching Pairs...")
    t0 = time.time()
    
    gt_df = con.execute(f"""
        WITH sample_gt AS (
            SELECT 
                source1_entity_id as s1_id,
                unnest(string_split(matched_entity_ids, ',')) as target_id
            FROM read_csv('{TRAIN_GT}', delim='\\t', header=True, quote='', all_varchar=True)
            WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
            USING SAMPLE 40000 (reservoir, 42)
        )
        SELECT 
            g.s1_id,
            g.target_id,
            substring(g.target_id, 1, 2) as target_source,
            s1.business_name as s1_name,
            s1.business_address as s1_addr,
            s1.country as country,
            COALESCE(s2.business_name, s3.business_name) as tgt_name,
            COALESCE(s2.business_address, s3.business_address) as tgt_addr,
            COALESCE(s2.country, s3.country) as tgt_country
        FROM sample_gt g
        JOIN read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True) s1 ON g.s1_id = s1.entity_id
        LEFT JOIN read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True) s2 ON g.target_id = s2.entity_id
        LEFT JOIN read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True) s3 ON g.target_id = s3.entity_id
    """).fetch_df()
    
    print(f"  Loaded {len(gt_df):,} true pairs. Computing features...")
    
    true_features = []
    for _, r in gt_df.iterrows():
        feats = compute_string_features(
            r['s1_name'], r['tgt_name'], 
            r['s1_addr'], r['tgt_addr'], 
            r['country'], r['tgt_country']
        )
        feats['s1_id'] = r['s1_id']
        feats['target_id'] = r['target_id']
        feats['target_source'] = r['target_source']
        feats['country'] = r['country']
        feats['is_match'] = 1
        true_features.append(feats)
        
    df_true = pd.DataFrame(true_features)
    true_parquet_path = os.path.join(REPORTS_DIR, "08_true_pair_features.parquet")
    df_true.to_parquet(true_parquet_path, index=False)
    print(f"  --> Saved {true_parquet_path}")

    # -------------------------------------------------------------
    # 2. Hard-Negative Analysis (Phase 7)
    # -------------------------------------------------------------
    print("\n[2/6] Mining Hard Negatives (H1 to H7)...")
    # Mine pairs sharing exact name or exact address or same postal that are NOT true matches
    # Register true pair set
    con.register('df_true_pairs', gt_df[['s1_id', 'target_id']])
    
    hard_neg_df = con.execute(f"""
        WITH targets_sample AS (
            SELECT entity_id, business_name, business_address, country, 'S2' as src
            FROM read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True)
            USING SAMPLE 200000 (reservoir, 42)
            UNION ALL
            SELECT entity_id, business_name, business_address, country, 'S3' as src
            FROM read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True)
            USING SAMPLE 200000 (reservoir, 42)
        ),
        s1_sample AS (
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True)
            USING SAMPLE 50000 (reservoir, 42)
        ),
        -- H1: Same normalized name, different entity
        h1_candidates AS (
            SELECT 
                s.entity_id as s1_id, t.entity_id as target_id, t.src as target_source,
                s.business_name as s1_name, s.business_address as s1_addr, s.country,
                t.business_name as tgt_name, t.business_address as tgt_addr, t.country as tgt_country,
                'H1_same_name_diff_entity' as hard_type
            FROM s1_sample s
            JOIN targets_sample t 
              ON lower(trim(regexp_replace(s.business_name, '[^a-zA-Z0-9]+', ' ', 'g'))) = 
                 lower(trim(regexp_replace(t.business_name, '[^a-zA-Z0-9]+', ' ', 'g')))
             AND s.country = t.country
            WHERE length(trim(s.business_name)) > 3
            LIMIT 10000
        ),
        -- H2: Same address, different entity
        h2_candidates AS (
            SELECT 
                s.entity_id as s1_id, t.entity_id as target_id, t.src as target_source,
                s.business_name as s1_name, s.business_address as s1_addr, s.country,
                t.business_name as tgt_name, t.business_address as tgt_addr, t.country as tgt_country,
                'H2_same_addr_diff_entity' as hard_type
            FROM s1_sample s
            JOIN targets_sample t 
              ON lower(trim(regexp_replace(s.business_address, '[^a-zA-Z0-9]+', ' ', 'g'))) = 
                 lower(trim(regexp_replace(t.business_address, '[^a-zA-Z0-9]+', ' ', 'g')))
             AND s.country = t.country
            WHERE length(trim(s.business_address)) > 6
            LIMIT 10000
        )
        SELECT * FROM h1_candidates
        UNION ALL
        SELECT * FROM h2_candidates
    """).fetch_df()
    
    # Filter out any accidental true matches
    gt_set = set(zip(df_true['s1_id'], df_true['target_id']))
    hard_neg_df['is_true'] = hard_neg_df.apply(lambda r: (r['s1_id'], r['target_id']) in gt_set, axis=1)
    hard_neg_df = hard_neg_df[~hard_neg_df['is_true']].copy()
    
    print(f"  Mined {len(hard_neg_df):,} hard negative pairs. Computing features...")
    
    hard_features = []
    for _, r in hard_neg_df.iterrows():
        feats = compute_string_features(
            r['s1_name'], r['tgt_name'], 
            r['s1_addr'], r['tgt_addr'], 
            r['country'], r['tgt_country']
        )
        feats['s1_id'] = r['s1_id']
        feats['target_id'] = r['target_id']
        feats['target_source'] = r['target_source']
        feats['country'] = r['country']
        feats['hard_type'] = r['hard_type']
        feats['is_match'] = 0
        hard_features.append(feats)
        
    df_hard = pd.DataFrame(hard_features)
    hard_parquet_path = os.path.join(REPORTS_DIR, "09_hard_negative_analysis.parquet")
    df_hard.to_parquet(hard_parquet_path, index=False)
    print(f"  --> Saved {hard_parquet_path}")

    # -------------------------------------------------------------
    # 3. Feature Separability Test (Phase 11)
    # -------------------------------------------------------------
    print("\n[3/6] Computing Feature Separability (ROC-AUC, PR-AUC, KS)...")
    # Combine true matches and negatives
    df_all = pd.concat([df_true, df_hard], ignore_index=True)
    
    feature_cols = [
        'name_exact', 'name_levenshtein', 'name_jaro_winkler', 'name_jaccard', 
        'name_token_overlap', 'name_length_ratio', 'both_addr_present',
        'addr_exact', 'addr_levenshtein', 'addr_jaro_winkler', 'addr_jaccard', 
        'postal_match', 'house_num_match', 'name_addr_agree', 'name_postal_agree'
    ]
    
    separability_rows = []
    for col in feature_cols:
        pos_vals = df_true[col].values
        neg_vals = df_hard[col].values
        
        # KS Statistic
        ks_stat, _ = ks_2samp(pos_vals, neg_vals)
        # ROC AUC
        roc_auc = roc_auc_score(df_all['is_match'], df_all[col]) if len(np.unique(df_all[col])) > 1 else 0.5
        # PR AUC
        pr_auc = average_precision_score(df_all['is_match'], df_all[col])
        
        separability_rows.append({
            'feature': col,
            'pos_mean': round(float(np.mean(pos_vals)), 4),
            'pos_median': round(float(np.median(pos_vals)), 4),
            'neg_mean': round(float(np.mean(neg_vals)), 4),
            'neg_median': round(float(np.median(neg_vals)), 4),
            'ks_statistic': round(float(ks_stat), 4),
            'roc_auc': round(float(roc_auc), 4),
            'pr_auc': round(float(pr_auc), 4)
        })
        
    df_sep = pd.DataFrame(separability_rows).sort_values(by='ks_statistic', ascending=False)
    sep_out = os.path.join(REPORTS_DIR, "10_feature_separability.csv")
    df_sep.to_csv(sep_out, index=False)
    print(f"  --> Saved {sep_out}")
    print(df_sep[['feature', 'pos_mean', 'neg_mean', 'ks_statistic', 'roc_auc', 'pr_auc']])

    # -------------------------------------------------------------
    # 4. Source-Pair Analysis (Phase 10: S1<->S2 vs S1<->S3)
    # -------------------------------------------------------------
    print("\n[4/6] Running Source-Pair Analysis (S1<->S2 vs S1<->S3)...")
    source_stats = []
    for src in ['S2', 'S3']:
        sub = df_true[df_true['target_source'] == src]
        source_stats.append({
            'source_pair': f"S1 <-> {src}",
            'pair_count': len(sub),
            'name_exact_rate': round(float(sub['name_exact'].mean()), 4),
            'name_levenshtein_mean': round(float(sub['name_levenshtein'].mean()), 4),
            'name_jaro_winkler_mean': round(float(sub['name_jaro_winkler'].mean()), 4),
            'addr_exact_rate': round(float(sub['addr_exact'].mean()), 4),
            'addr_levenshtein_mean': round(float(sub['addr_levenshtein'].mean()), 4),
            'postal_match_rate': round(float(sub['postal_match'].mean()), 4),
            'both_addr_present_rate': round(float(sub['both_addr_present'].mean()), 4)
        })
    df_src = pd.DataFrame(source_stats)
    src_out = os.path.join(REPORTS_DIR, "11_source_pair_analysis.csv")
    df_src.to_csv(src_out, index=False)
    print(f"  --> Saved {src_out}")
    print(df_src)

    # -------------------------------------------------------------
    # 5. Country Analysis (Phase 9: US vs India)
    # -------------------------------------------------------------
    print("\n[5/6] Running Country Analysis (US vs India)...")
    country_stats = []
    for cntry in ['US', 'India']:
        sub = df_true[df_true['country'] == cntry]
        country_stats.append({
            'country': cntry,
            'pair_count': len(sub),
            'name_exact_rate': round(float(sub['name_exact'].mean()), 4),
            'name_levenshtein_mean': round(float(sub['name_levenshtein'].mean()), 4),
            'addr_exact_rate': round(float(sub['addr_exact'].mean()), 4),
            'addr_levenshtein_mean': round(float(sub['addr_levenshtein'].mean()), 4),
            'postal_match_rate': round(float(sub['postal_match'].mean()), 4),
            'both_addr_present_rate': round(float(sub['both_addr_present'].mean()), 4)
        })
    df_cntry = pd.DataFrame(country_stats)
    cntry_out = os.path.join(REPORTS_DIR, "12_country_analysis.csv")
    df_cntry.to_csv(cntry_out, index=False)
    print(f"  --> Saved {cntry_out}")
    print(df_cntry)

    # -------------------------------------------------------------
    # 6. Missingness Analysis (Phase 8: Address available vs missing)
    # -------------------------------------------------------------
    print("\n[6/6] Running Missingness Analysis...")
    missing_stats = []
    for status, mask in [('Address_Present', df_true['both_addr_present'] == 1.0),
                         ('Address_Missing', df_true['both_addr_present'] == 0.0)]:
        sub = df_true[mask]
        missing_stats.append({
            'status': status,
            'pair_count': len(sub),
            'pct_of_true_pairs': round(float(len(sub) / len(df_true) * 100), 2),
            'name_exact_rate': round(float(sub['name_exact'].mean()), 4),
            'name_levenshtein_mean': round(float(sub['name_levenshtein'].mean()), 4),
            'name_jaro_winkler_mean': round(float(sub['name_jaro_winkler'].mean()), 4),
            'name_jaccard_mean': round(float(sub['name_jaccard'].mean()), 4)
        })
    df_miss = pd.DataFrame(missing_stats)
    miss_out = os.path.join(REPORTS_DIR, "13_missingness_analysis.csv")
    df_miss.to_csv(miss_out, index=False)
    print(f"  --> Saved {miss_out}")
    print(df_miss)
    print("\nStep 4 Complete!")

if __name__ == "__main__":
    run_step04()
