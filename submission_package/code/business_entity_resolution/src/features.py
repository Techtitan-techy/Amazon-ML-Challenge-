"""
Pairwise Feature Extraction Layer for Business Entity Resolution.
Amazon ML Challenge 2026.

Supports:
1. Production 15 SIMD RapidFuzz features (Address-Availability-Aware)
   for 100% backward compatibility with frozen matcher.
2. Championship 35-Dimensional Feature Matrix (Stage 4 & Stage 8)
   incorporating lexical, structural, numeric, retrieval consensus,
   and collective competition signals.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein

# Base 15 production features
FEATURE_NAMES = [
    "addr_jaccard",
    "addr_jaro_winkler",
    "addr_levenshtein",
    "both_addr_present",
    "name_jaro_winkler",
    "name_levenshtein",
    "name_jaccard",
    "house_num_match",
    "postal_match",
    "addr_missing",
    "name_x_addr_present",
    "name_x_addr_missing",
    "addr_sim_x_addr_present",
    "house_num_x_addr_present",
    "postal_x_addr_present"
]

FEATURE_NAMES_18 = [
    'name_exact', 'name_levenshtein', 'name_jaro_winkler', 'name_jaccard',
    'name_token_overlap', 'name_length_ratio', 'has_addr_s1', 'has_addr_tgt',
    'both_addr_present', 'addr_exact', 'addr_levenshtein', 'addr_jaro_winkler',
    'addr_jaccard', 'postal_match', 'house_num_match', 'name_addr_agree',
    'name_postal_agree', 'retrieval_score'
]


def extract_18_features(s1_rec: dict, target_rec: dict, ret_score: float = 0.90) -> list:
    """Extract 18 features matching the pre-trained XGBoost matcher."""
    n1 = s1_rec.get("norm_name") or s1_rec.get("business_name") or ""
    n2 = target_rec.get("norm_name") or target_rec.get("business_name") or ""
    a1 = s1_rec.get("norm_addr") or s1_rec.get("business_address") or ""
    a2 = target_rec.get("norm_addr") or target_rec.get("business_address") or ""
    
    n_exact = 1.0 if (n1 and n1 == n2) else 0.0
    n_lev = Levenshtein.normalized_similarity(n1, n2) if (n1 and n2) else 0.0
    n_jw = JaroWinkler.similarity(n1, n2) if (n1 and n2) else 0.0
    
    s1_toks = set(n1.split())
    tgt_toks = set(n2.split())
    inter = len(s1_toks & tgt_toks)
    union = len(s1_toks | tgt_toks)
    n_jaccard = inter / union if union > 0 else 0.0
    n_overlap = inter / min(len(s1_toks), len(tgt_toks)) if min(len(s1_toks), len(tgt_toks)) > 0 else 0.0
    n_len_ratio = min(len(n1), len(n2)) / max(len(n1), len(n2)) if max(len(n1), len(n2)) > 0 else 0.0
    
    has_addr_s1 = 1.0 if a1 else 0.0
    has_addr_tgt = 1.0 if a2 else 0.0
    both_addr = 1.0 if (has_addr_s1 and has_addr_tgt) else 0.0
    
    a_exact = 1.0 if (both_addr and a1 == a2) else 0.0
    a_lev = Levenshtein.normalized_similarity(a1, a2) if both_addr else 0.0
    a_jw = JaroWinkler.similarity(a1, a2) if both_addr else 0.0
    
    a1_toks = set(a1.split())
    a2_toks = set(a2.split())
    a_inter = len(a1_toks & a2_toks)
    a_union = len(a1_toks | a2_toks)
    a_jaccard = a_inter / a_union if a_union > 0 else 0.0
    
    p1 = s1_rec.get("postal_code") or s1_rec.get("postal") or ""
    p2 = target_rec.get("postal_code") or target_rec.get("postal") or ""
    postal_match = 1.0 if (p1 and p2 and p1 == p2) else 0.0
    
    h1 = s1_rec.get("house_number") or s1_rec.get("house_num") or ""
    h2 = target_rec.get("house_number") or target_rec.get("house_num") or ""
    house_match = 1.0 if (h1 and h2 and h1 == h2) else 0.0
    
    name_addr_agree = 1.0 if (n_exact == 1.0 and a_exact == 1.0) else 0.0
    name_postal_agree = 1.0 if (n_exact == 1.0 and postal_match == 1.0) else 0.0
    
    return [
        n_exact, n_lev, n_jw, n_jaccard, n_overlap, n_len_ratio,
        has_addr_s1, has_addr_tgt, both_addr, a_exact, a_lev, a_jw, a_jaccard,
        postal_match, house_match, name_addr_agree, name_postal_agree, float(ret_score)
    ]


# Championship 42-feature matrix (incorporating best signals from Akash-bardia & Swaindhruti)
CHAMPIONSHIP_FEATURE_NAMES = FEATURE_NAMES + [
    "name_exact",
    "name_containment",
    "name_token_overlap",
    "name_length_ratio",
    "name_token_diff",
    "name_char3_jaccard",
    "name_phonetic_sim",
    "addr_token_overlap",
    "addr_char3_jaccard",
    "city_overlap",
    "numeric_overlap",
    "name_strong_addr_weak",
    "name_weak_addr_strong",
    "target_source_s3",
    "retrieval_score",
    "retrieval_channel_count",
    "cand_rank",
    "margin_to_top",
    "margin_to_next",
    "pool_score_density",
    # Competitor Surpassing Signals (Akash-bardia & Swaindhruti / aads)
    "first_token_match",
    "first_token_conflict",
    "primary_num_mismatch",
    "name_high_but_num_conflict",
    "cross_prod",
    "name_char3_cos",
    "addr_char3_cos"
]


def _char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _char_ngram_cos(s1: str, s2: str, n: int = 3) -> float:
    """Character n-gram cosine similarity (Swaindhruti / Team aads innovation)."""
    if not s1 or not s2:
        return 0.0
    l1 = len(s1)
    l2 = len(s2)
    if l1 < n or l2 < n:
        return 1.0 if s1 == s2 else 0.0
    c1 = _char_ngrams(s1, n)
    c2 = _char_ngrams(s2, n)
    if not c1 or not c2:
        return 0.0
    intersection = len(c1 & c2)
    norm = np.sqrt(len(c1)) * np.sqrt(len(c2))
    return float(intersection / norm) if norm > 0 else 0.0


def extract_pair_features(s1_rec: Dict[str, Any], target_rec: Dict[str, Any]) -> List[float]:
    """
    Extract the 15 verified address-availability-aware pairwise features.
    Maintains 100% exact compatibility with production baseline.
    """
    n1 = s1_rec.get("norm_name") or s1_rec.get("business_name") or ""
    n2 = target_rec.get("norm_name") or target_rec.get("business_name") or ""
    a1 = s1_rec.get("norm_addr") or s1_rec.get("business_address") or ""
    a2 = target_rec.get("norm_addr") or target_rec.get("business_address") or ""

    # 1. Name features
    name_jw = JaroWinkler.normalized_similarity(n1, n2)
    name_lev = Levenshtein.normalized_similarity(n1, n2)

    nt1 = set(n1.split())
    nt2 = set(n2.split())
    name_jaccard = len(nt1 & nt2) / len(nt1 | nt2) if (nt1 | nt2) else 0.0

    # 2. Address features & evidence regimes
    both_addr = 1.0 if (len(a1) > 0 and len(a2) > 0) else 0.0
    addr_missing = 1.0 - both_addr

    if both_addr == 1.0:
        at1 = set(a1.split())
        at2 = set(a2.split())
        addr_jaccard = len(at1 & at2) / len(at1 | at2) if (at1 | at2) else 0.0
        addr_jw = JaroWinkler.normalized_similarity(a1, a2)
        addr_lev = Levenshtein.normalized_similarity(a1, a2)

        # Postal match (digits of len 5 or 6)
        post1 = [w for w in at1 if len(w) in (5, 6) and w.isdigit()]
        post2 = [w for w in at2 if len(w) in (5, 6) and w.isdigit()]
        postal_match = 1.0 if (post1 and post2 and set(post1) & set(post2)) else 0.0

        # House / Building number match
        num1 = [w for w in at1 if any(c.isdigit() for c in w)]
        num2 = [w for w in at2 if any(c.isdigit() for c in w)]
        house_num_match = 1.0 if (num1 and num2 and set(num1) & set(num2)) else 0.0
    else:
        addr_jaccard = 0.0
        addr_jw = 0.0
        addr_lev = 0.0
        postal_match = 0.0
        house_num_match = 0.0

    # 3. Address-Availability-Aware Interactions
    name_x_addr_present = name_jw * both_addr
    name_x_addr_missing = name_jw * addr_missing
    addr_sim_x_addr_present = addr_jw * both_addr
    house_num_x_addr_present = house_num_match * both_addr
    postal_x_addr_present = postal_match * both_addr

    return [
        addr_jaccard,
        addr_jw,
        addr_lev,
        both_addr,
        name_jw,
        name_lev,
        name_jaccard,
        house_num_match,
        postal_match,
        addr_missing,
        name_x_addr_present,
        name_x_addr_missing,
        addr_sim_x_addr_present,
        house_num_x_addr_present,
        postal_x_addr_present
    ]


def extract_championship_features(
    s1_rec: Dict[str, Any],
    target_rec: Dict[str, Any],
    retrieval_meta: Optional[Dict[str, Any]] = None,
    collective_meta: Optional[Dict[str, Any]] = None
) -> List[float]:
    """
    Extract the complete 35-dimensional championship feature matrix:
    - Base 15 features
    - Lexical containment, character 3-grams, and phonetic similarity
    - Numeric overlap and city token overlap
    - Source flag (S2 vs S3)
    - Retrieval confidence and channel hits
    - Collective/sibling features (candidate rank, margins, pool density)
    """
    base_feats = extract_pair_features(s1_rec, target_rec)

    n1 = s1_rec.get("norm_name") or s1_rec.get("business_name") or ""
    n2 = target_rec.get("norm_name") or target_rec.get("business_name") or ""
    a1 = s1_rec.get("norm_addr") or s1_rec.get("business_address") or ""
    a2 = target_rec.get("norm_addr") or target_rec.get("business_address") or ""

    nt1 = n1.split()
    nt2 = n2.split()
    s_nt1 = set(nt1)
    s_nt2 = set(nt2)

    # 1. Lexical enhancements
    name_exact = 1.0 if (n1 == n2 and len(n1) > 0) else 0.0
    name_containment = 1.0 if (n1 and n2 and (n1 in n2 or n2 in n1)) else 0.0
    min_tokens = min(len(s_nt1), len(s_nt2))
    name_token_overlap = len(s_nt1 & s_nt2) / min_tokens if min_tokens > 0 else 0.0

    max_len = max(len(n1), len(n2))
    name_length_ratio = min(len(n1), len(n2)) / max_len if max_len > 0 else 0.0
    name_token_diff = float(abs(len(nt1) - len(nt2)))

    # 2. Character n-gram Jaccard (typos / OCR transpositions)
    c1 = _char_ngrams(n1, 3)
    c2 = _char_ngrams(n2, 3)
    name_char3_jaccard = len(c1 & c2) / len(c1 | c2) if (c1 | c2) else 0.0

    # 3. Phonetic similarity
    sk1 = s1_rec.get("phonetic_skeleton") or ""
    sk2 = target_rec.get("phonetic_skeleton") or ""
    name_phonetic_sim = JaroWinkler.normalized_similarity(sk1, sk2) if (sk1 and sk2) else 0.0

    # 4. Address enhancements
    at1 = a1.split()
    at2 = a2.split()
    s_at1 = set(at1)
    s_at2 = set(at2)
    min_at = min(len(s_at1), len(s_at2))
    addr_token_overlap = len(s_at1 & s_at2) / min_at if min_at > 0 else 0.0

    ac1 = _char_ngrams(a1, 3)
    ac2 = _char_ngrams(a2, 3)
    addr_char3_jaccard = len(ac1 & ac2) / len(ac1 | ac2) if (ac1 | ac2) else 0.0

    # City / locality tokens
    loc1 = set(at1[-3:]) if len(at1) >= 2 else s_at1
    loc2 = set(at2[-3:]) if len(at2) >= 2 else s_at2
    city_overlap = len(loc1 & loc2) / len(loc1 | loc2) if (loc1 | loc2) else 0.0

    # Numeric tokens overlap
    nums1 = s1_rec.get("numeric_tokens") or {w for w in (nt1 + at1) if any(c.isdigit() for c in w)}
    nums2 = target_rec.get("numeric_tokens") or {w for w in (nt2 + at2) if any(c.isdigit() for c in w)}
    numeric_overlap = len(nums1 & nums2) / len(nums1 | nums2) if (nums1 | nums2) else 0.0

    # Cross-field anomalies
    name_jw = base_feats[4]
    addr_jw = base_feats[1]
    both_addr = base_feats[3]
    name_strong_addr_weak = 1.0 if (name_jw > 0.88 and addr_jw < 0.35 and both_addr == 1.0) else 0.0
    name_weak_addr_strong = 1.0 if (name_jw < 0.60 and addr_jw > 0.85 and both_addr == 1.0) else 0.0

    # Target source (S2=0, S3=1)
    target_id = str(target_rec.get("entity_id") or "")
    target_source_s3 = 1.0 if target_id.startswith("S3-") else 0.0

    # Retrieval meta
    rm = retrieval_meta or {}
    retrieval_score = float(rm.get("retrieval_score", 0.0))
    retrieval_channel_count = float(rm.get("channel_hits", 1.0))

    # Collective / sibling competition meta
    cm = collective_meta or {}
    cand_rank = float(cm.get("rank", 0.0))
    margin_to_top = float(cm.get("margin_to_top", 0.0))
    margin_to_next = float(cm.get("margin_to_next", 0.0))
    pool_score_density = float(cm.get("pool_density", 0.5))

    # Competitor Surpassing Signals (Akash-bardia & Swaindhruti / Team aads)
    first_token_match = 1.0 if (nt1 and nt2 and nt1[0] == nt2[0]) else 0.0
    first_token_conflict = 1.0 if (nt1 and nt2 and nt1[0] != nt2[0]) else 0.0

    house1 = s1_rec.get("house_number") or ""
    house2 = target_rec.get("house_number") or ""
    primary_num_mismatch = 1.0 if (house1 and house2 and house1 != house2) else 0.0
    name_high_but_num_conflict = 1.0 if (name_jw > 0.85 and primary_num_mismatch == 1.0) else 0.0

    cross_prod = float(name_jw * addr_jw)
    name_char3_cos = _char_ngram_cos(n1, n2, 3)
    addr_char3_cos = _char_ngram_cos(a1, a2, 3)

    extended_feats = [
        name_exact,
        name_containment,
        name_token_overlap,
        name_length_ratio,
        name_token_diff,
        name_char3_jaccard,
        name_phonetic_sim,
        addr_token_overlap,
        addr_char3_jaccard,
        city_overlap,
        numeric_overlap,
        name_strong_addr_weak,
        name_weak_addr_strong,
        target_source_s3,
        retrieval_score,
        retrieval_channel_count,
        cand_rank,
        margin_to_top,
        margin_to_next,
        pool_score_density,
        # Competitor Surpassing Signals
        first_token_match,
        first_token_conflict,
        primary_num_mismatch,
        name_high_but_num_conflict,
        cross_prod,
        name_char3_cos,
        addr_char3_cos
    ]

    return base_feats + extended_feats
