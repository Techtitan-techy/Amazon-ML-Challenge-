import os
import sys
import json
import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REPORTS_DIR
from normalization import transliterated

def run_step09():
    print("=" * 70)
    print("DEEP-DIVE DIAGNOSTIC ON THE 476 MISSED GROUND TRUTH PAIRS")
    print("=" * 70)
    
    missed_csv = os.path.join(REPORTS_DIR, "19_missed_gt_analysis.csv")
    if not os.path.exists(missed_csv):
        print(f"Error: {missed_csv} not found.")
        return
        
    df_missed = pd.read_csv(missed_csv)
    total_missed = len(df_missed)
    print(f"Analyzing all {total_missed} missed ground truth pairs...")
    
    categories = []
    category_counts = {
        "missing_target_address": 0,
        "state_acronym_or_short_address": 0,
        "indic_phonetic_distance": 0,
        "name_acronym_or_extreme_truncation": 0,
        "address_token_displacement": 0,
        "severe_name_typo_or_corruption": 0,
        "other_residual": 0
    }
    
    # Common Indian state acronyms in addresses
    state_acronyms = {'wb', 'mh', 'mp', 'up', 'dl', 'ka', 'tn', 'ts', 'ap', 'hr', 'pb', 'rj', 'gj', 'kl'}
    
    classified_rows = []
    
    for idx, r in df_missed.iterrows():
        s1_n = transliterated(r['s1_name'])
        tgt_n = transliterated(r['tgt_name'])
        s1_a = transliterated(r['s1_addr'])
        tgt_a = transliterated(r['tgt_addr'])
        
        # Similarities
        n_lev = Levenshtein.normalized_similarity(s1_n, tgt_n) if (s1_n and tgt_n) else 0.0
        n_jw = JaroWinkler.similarity(s1_n, tgt_n) if (s1_n and tgt_n) else 0.0
        
        s1_n_toks = set(s1_n.split())
        tgt_n_toks = set(tgt_n.split())
        n_jaccard = len(s1_n_toks & tgt_n_toks) / len(s1_n_toks | tgt_n_toks) if (s1_n_toks and tgt_n_toks) else 0.0
        
        a_lev = Levenshtein.normalized_similarity(s1_a, tgt_a) if (s1_a and tgt_a) else 0.0
        s1_a_toks = set(s1_a.split())
        tgt_a_toks = set(tgt_a.split())
        a_jaccard = len(s1_a_toks & tgt_a_toks) / len(s1_a_toks | tgt_a_toks) if (s1_a_toks and tgt_a_toks) else 0.0
        
        # Classify root cause
        tgt_addr_missing = bool(r['tgt_addr_missing']) or not tgt_a
        
        # Check acronym
        is_acronym = False
        if len(s1_n.split()) > 1 and len(tgt_n.split()) == 1 and len(tgt_n) <= 5:
            initials = "".join([w[0] for w in s1_n.split() if w])
            if tgt_n in initials or initials in tgt_n:
                is_acronym = True
        elif len(tgt_n.split()) > 1 and len(s1_n.split()) == 1 and len(s1_n) <= 5:
            initials = "".join([w[0] for w in tgt_n.split() if w])
            if s1_n in initials or initials in s1_n:
                is_acronym = True
                
        # Check state acronym in target address
        has_state_acronym = bool(state_acronyms.intersection(tgt_a_toks))
        
        # Check Indic phonetic distance (e.g. jw > 0.70 but jaccard == 0 due to phonetic transliteration)
        is_indic_phonetic = (r['country'] == 'India') and (n_jw >= 0.75) and (n_jaccard < 0.20)
        
        if tgt_addr_missing:
            cause = "missing_target_address"
        elif is_acronym or (len(s1_n) / (len(tgt_n)+1) > 2.5) or (len(tgt_n) / (len(s1_n)+1) > 2.5):
            cause = "name_acronym_or_extreme_truncation"
        elif has_state_acronym and len(tgt_a_toks) <= 5:
            cause = "state_acronym_or_short_address"
        elif is_indic_phonetic:
            cause = "indic_phonetic_distance"
        elif a_jaccard < 0.15 and n_jw >= 0.75:
            cause = "address_token_displacement"
        elif n_jw < 0.70:
            cause = "severe_name_typo_or_corruption"
        else:
            cause = "other_residual"
            
        category_counts[cause] += 1
        
        classified_rows.append({
            's1_id': r['s1_id'],
            'target_id': r['target_id'],
            'country': r['country'],
            's1_name': r['s1_name'],
            'tgt_name': r['tgt_name'],
            's1_addr': r['s1_addr'],
            'tgt_addr': r['tgt_addr'],
            'name_levenshtein': round(n_lev, 4),
            'name_jaro_winkler': round(n_jw, 4),
            'name_jaccard': round(n_jaccard, 4),
            'addr_levenshtein': round(a_lev, 4),
            'addr_jaccard': round(a_jaccard, 4),
            'primary_root_cause': cause
        })
        
    df_classified = pd.DataFrame(classified_rows)
    deepdive_out = os.path.join(REPORTS_DIR, "24_missed_gt_deepdive.csv")
    df_classified.to_csv(deepdive_out, index=False)
    print(f"  --> Saved {deepdive_out}")
    
    # Summary Table
    breakdown = []
    for cause, cnt in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = round(cnt / total_missed * 100, 2)
        breakdown.append({
            "root_cause": cause,
            "count": cnt,
            "percentage_of_missed": pct,
            "pct_of_all_eval_gt": round(cnt / 68963 * 100, 4)
        })
        print(f"  {cause:35s}: {cnt:>4d} pairs ({pct:>5.2f}% of missed | {cnt/68963*100:.3f}% of total GT)")
        
    df_summary = pd.DataFrame(breakdown)
    breakdown_json = os.path.join(REPORTS_DIR, "25_missed_gt_breakdown.json")
    with open(breakdown_json, "w", encoding="utf-8") as f:
        json.dump(breakdown, f, indent=2)
    print(f"  --> Saved {breakdown_json}")
    
    print("\nRoot Cause Breakdown Complete!")

if __name__ == "__main__":
    run_step09()
