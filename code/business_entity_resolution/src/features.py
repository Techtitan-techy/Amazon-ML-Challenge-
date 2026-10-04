"""
Business Entity Resolution — SIMD Pairwise Feature Engineering
==============================================================
Computes highly discriminative pairwise similarity features between Source 1
and Candidate (Source 2 / Source 3) entity pairs using C++ SIMD-accelerated
RapidFuzz string algorithms, token set overlaps, and structured field matches.
"""

import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler

# Ordered list of feature names matching model training
FEATURE_NAMES = [
    'name_exact',
    'name_levenshtein',
    'name_jaro_winkler',
    'name_jaccard',
    'name_token_overlap',
    'name_length_ratio',
    'has_addr_s1',
    'has_addr_tgt',
    'both_addr_present',
    'addr_exact',
    'addr_levenshtein',
    'addr_jaro_winkler',
    'addr_jaccard',
    'postal_match',
    'house_num_match',
    'name_addr_agree',
    'name_postal_agree',
    'retrieval_score',
]


def compute_pair_feature_vector(s1_name: str, tgt_name: str,
                                s1_addr: str, tgt_addr: str,
                                s1_postal: str, tgt_postal: str,
                                s1_house: str, tgt_house: str,
                                ret_score: float) -> list:
    """Compute single-pair feature list in exact FEATURE_NAMES order."""
    # 1. Name features
    n_exact = 1.0 if (s1_name and s1_name == tgt_name) else 0.0
    n_lev = Levenshtein.normalized_similarity(s1_name, tgt_name) if (s1_name and tgt_name) else 0.0
    n_jw = JaroWinkler.similarity(s1_name, tgt_name) if (s1_name and tgt_name) else 0.0
    
    s1_toks = set(s1_name.split())
    tgt_toks = set(tgt_name.split())
    n_overlap_count = len(s1_toks & tgt_toks)
    n_jaccard = n_overlap_count / len(s1_toks | tgt_toks) if (s1_toks and tgt_toks) else 0.0
    n_overlap = n_overlap_count / min(len(s1_toks), len(tgt_toks)) if (s1_toks and tgt_toks) else 0.0
    n_len_ratio = min(len(s1_name), len(tgt_name)) / max(len(s1_name), len(tgt_name)) if max(len(s1_name), len(tgt_name)) > 0 else 0.0
    
    # 2. Address features
    has_addr_s1 = 1.0 if s1_addr else 0.0
    has_addr_tgt = 1.0 if tgt_addr else 0.0
    both_addr_present = 1.0 if (has_addr_s1 and has_addr_tgt) else 0.0
    
    a_exact = 1.0 if (both_addr_present and s1_addr == tgt_addr) else 0.0
    a_lev = Levenshtein.normalized_similarity(s1_addr, tgt_addr) if both_addr_present else 0.0
    a_jw = JaroWinkler.similarity(s1_addr, tgt_addr) if both_addr_present else 0.0
    
    a1_toks = set(s1_addr.split())
    a2_toks = set(tgt_addr.split())
    a_jaccard = len(a1_toks & a2_toks) / len(a1_toks | a2_toks) if (a1_toks and a2_toks) else 0.0
    
    # 3. Structured / Agreement features
    postal_match = 1.0 if (s1_postal and tgt_postal and s1_postal == tgt_postal) else 0.0
    house_match = 1.0 if (s1_house and tgt_house and s1_house == tgt_house) else 0.0
    name_addr_agree = 1.0 if (n_exact == 1.0 and a_exact == 1.0) else 0.0
    name_postal_agree = 1.0 if (n_exact == 1.0 and postal_match == 1.0) else 0.0
    
    return [
        n_exact,
        n_lev,
        n_jw,
        n_jaccard,
        n_overlap,
        n_len_ratio,
        has_addr_s1,
        has_addr_tgt,
        both_addr_present,
        a_exact,
        a_lev,
        a_jw,
        a_jaccard,
        postal_match,
        house_match,
        name_addr_agree,
        name_postal_agree,
        float(ret_score)
    ]


def compute_batch_features(pairs_data: list) -> np.ndarray:
    """
    Compute pairwise features for a batch of candidate pairs.
    Each item in pairs_data is a tuple/list:
    (s1_name, tgt_name, s1_addr, tgt_addr, s1_postal, tgt_postal, s1_house, tgt_house, ret_score)
    Returns:
        np.ndarray of shape (N, len(FEATURE_NAMES)) with dtype float32.
    """
    n = len(pairs_data)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        
    feat_matrix = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    for i, item in enumerate(pairs_data):
        feat_matrix[i, :] = compute_pair_feature_vector(*item)
        
    return feat_matrix
