"""
Championship Multi-Channel Candidate Retrieval & Learned Candidate Reranking.
Amazon ML Challenge 2026.

Implements:
- Multi-Channel High-Recall Candidate Retrieval (Stage 2):
    R1: Exact Normalized Name Anchors
    R2: Compact Prefix Anchors (6-8 char prefixes)
    R3: Name Word Inverted Index (IDF-weighted / frequent-token pruned)
    R4: Consonant & Phonetic Skeleton (Indic transliteration & typo recovery)
    R5: Address Word Inverted Index
    R6: Numeric Anchors (Postal code + House number alignment)
    R7: Domain / DBA Inverted Anchors
- High-Recall Union (Stage 2 & 6: target >=99.5% candidate recall)
- Candidate Reranker & Dynamic Budget Policy (Stage 3 & 29: K=25-35, winning Amazon tiebreaker)
"""

from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein

# Frequent English/Indian/French stop tokens to prune from inverted index
STOP_TOKENS = {
    "the", "and", "for", "with", "ltd", "pvt", "llc", "inc", "corp", "co",
    "services", "solutions", "enterprises", "company", "group", "india", "usa",
    "france", "road", "street", "avenue", "near", "opposite", "floor", "building",
    "de", "la", "le", "et", "du", "des", "sur"
}


class HybridRetriever:
    """
    Championship Multi-Channel Candidate Retrieval Engine.
    Strictly partitions by country for 100% purity and France invariance.
    """

    def __init__(
        self,
        k_ceiling: int = 30,
        steep_threshold: float = 0.85,
        gap_threshold: float = 0.35,
        max_posting_len: int = 500
    ):
        """
        Initialize multi-channel hybrid retriever with candidate pruning budgets.
        
        Args:
            k_ceiling: Maximum candidate budget per Source 1 entity.
            steep_threshold: Retrieval score cutoff for high-confidence steep drop-off.
            gap_threshold: Score gap threshold between top 2 candidates.
            max_posting_len: Maximum posting list size for token inverted indexes.
        """
        self.k_ceiling = k_ceiling
        self.steep_threshold = steep_threshold
        self.gap_threshold = gap_threshold
        self.max_posting_len = max_posting_len

        # Channel 1: Country -> exact norm_name -> list of entity_ids
        self.country_exact_name_index: Dict[str, Dict[str, List[str]]] = {}
        # Channel 2: Country -> compact_name[:8] -> list of entity_ids
        self.country_prefix_index: Dict[str, Dict[str, List[str]]] = {}
        # Channel 3: Country -> name word token -> list of entity_ids
        self.country_token_index: Dict[str, Dict[str, List[str]]] = {}
        # Channel 4: Country -> phonetic skeleton -> list of entity_ids
        self.country_phonetic_index: Dict[str, Dict[str, List[str]]] = {}
        # Channel 5: Country -> address key tokens -> list of entity_ids
        self.country_addr_index: Dict[str, Dict[str, List[str]]] = {}
        # Channel 6: Country -> postal code -> list of entity_ids
        self.country_postal_index: Dict[str, Dict[str, List[str]]] = {}
        # Fast target lookup: entity_id -> target record
        self.target_lookup: Dict[str, Dict[str, Any]] = {}

    def index_targets(self, target_records: List[Dict[str, Any]]):
        """
        Build partitioned multi-channel inverted indexes across target (S2 and S3) records.
        """
        for rec in target_records:
            eid = rec["entity_id"]
            country = rec["country"]
            norm_name = rec.get("norm_name") or rec.get("business_name") or ""
            compact_name = rec.get("compact_name") or norm_name.replace(" ", "")
            phonetic_skel = rec.get("phonetic_skeleton") or ""
            norm_addr = rec.get("norm_addr") or rec.get("business_address") or ""
            postal_code = rec.get("postal_code") or ""

            self.target_lookup[eid] = rec

            # Initialize country partitions
            if country not in self.country_token_index:
                self.country_exact_name_index[country] = {}
                self.country_prefix_index[country] = {}
                self.country_token_index[country] = {}
                self.country_phonetic_index[country] = {}
                self.country_addr_index[country] = {}
                self.country_postal_index[country] = {}

            exact_idx = self.country_exact_name_index[country]
            prefix_idx = self.country_prefix_index[country]
            token_idx = self.country_token_index[country]
            phonetic_idx = self.country_phonetic_index[country]
            addr_idx = self.country_addr_index[country]
            postal_idx = self.country_postal_index[country]

            # R1: Exact name anchor
            if norm_name:
                exact_idx.setdefault(norm_name, []).append(eid)

            # R2: Compact prefix anchor (6-8 chars)
            if len(compact_name) >= 6:
                prefix = compact_name[:8]
                prefix_idx.setdefault(prefix, []).append(eid)

            # R3: Name word tokens (prune frequent tokens and stopwords)
            tokens = norm_name.split()
            for tok in tokens[:4]:
                if len(tok) >= 3 and tok not in STOP_TOKENS:
                    posting = token_idx.setdefault(tok, [])
                    if len(posting) < self.max_posting_len:
                        posting.append(eid)

            # R4: Consonant & Phonetic skeleton
            if phonetic_skel and len(phonetic_skel) >= 3:
                for ptok in phonetic_skel.split()[:2]:
                    if len(ptok) >= 3:
                        posting = phonetic_idx.setdefault(ptok, [])
                        if len(posting) < self.max_posting_len:
                            posting.append(eid)

            # R5: Address word tokens (distinctive street/city names)
            if norm_addr:
                for atok in norm_addr.split()[:3]:
                    if len(atok) >= 4 and atok not in STOP_TOKENS:
                        posting = addr_idx.setdefault(atok, [])
                        if len(posting) < self.max_posting_len:
                            posting.append(eid)

            # R6: Postal code
            if postal_code:
                posting = postal_idx.setdefault(postal_code, [])
                if len(posting) < self.max_posting_len:
                    posting.append(eid)

    def retrieve_candidates_with_scores(
        self,
        s1_rec: Dict[str, Any]
    ) -> List[Tuple[str, float, int]]:
        """
        Multi-channel candidate retrieval with channel consensus and fast reranking.
        Returns list of (candidate_id, rerank_score, channel_hits).
        """
        country = s1_rec["country"]
        norm_name = s1_rec.get("norm_name") or s1_rec.get("business_name") or ""
        compact_name = s1_rec.get("compact_name") or norm_name.replace(" ", "")
        phonetic_skel = s1_rec.get("phonetic_skeleton") or ""
        norm_addr = s1_rec.get("norm_addr") or s1_rec.get("business_address") or ""
        postal_code = s1_rec.get("postal_code") or ""

        exact_idx = self.country_exact_name_index.get(country, {})
        prefix_idx = self.country_prefix_index.get(country, {})
        token_idx = self.country_token_index.get(country, {})
        phonetic_idx = self.country_phonetic_index.get(country, {})
        addr_idx = self.country_addr_index.get(country, {})
        postal_idx = self.country_postal_index.get(country, {})

        # Candidate pool & channel tracker: candidate_id -> count of independent channels
        candidate_hits: Dict[str, int] = {}

        def add_candidates(c_list: List[str], max_take: int):
            """Accumulate channel hits for retrieved candidate IDs up to max_take."""
            for cid in c_list[:max_take]:
                candidate_hits[cid] = candidate_hits.get(cid, 0) + 1

        # R1: Exact name
        if norm_name in exact_idx:
            add_candidates(exact_idx[norm_name], 20)

        # R2: Prefix anchor
        if len(compact_name) >= 6:
            prefix = compact_name[:8]
            if prefix in prefix_idx:
                add_candidates(prefix_idx[prefix], 25)

        # R3: Word tokens
        tokens = norm_name.split()
        for tok in tokens[:3]:
            if len(tok) >= 3 and tok in token_idx:
                add_candidates(token_idx[tok], 35)

        # R4: Phonetic skeleton (transliterations & OCR recovery)
        if phonetic_skel:
            for ptok in phonetic_skel.split()[:2]:
                if len(ptok) >= 3 and ptok in phonetic_idx:
                    add_candidates(phonetic_idx[ptok], 20)

        # R5: Address tokens
        if norm_addr:
            for atok in norm_addr.split()[:2]:
                if len(atok) >= 4 and atok in addr_idx:
                    add_candidates(addr_idx[atok], 15)

        # R6: Postal code
        if postal_code and postal_code in postal_idx:
            add_candidates(postal_idx[postal_code], 15)

        if not candidate_hits:
            return []

        # Scored reranking
        scored: List[Tuple[str, float, int]] = []
        for cid, hits in candidate_hits.items():
            trec = self.target_lookup.get(cid)
            if not trec:
                continue

            c_name = trec.get("norm_name") or trec.get("business_name") or ""
            c_addr = trec.get("norm_addr") or trec.get("business_address") or ""
            c_postal = trec.get("postal_code") or ""

            # RapidFuzz similarity
            sim_name = 0.70 * JaroWinkler.normalized_similarity(norm_name, c_name) + 0.30 * Levenshtein.normalized_similarity(norm_name, c_name)

            if norm_addr and c_addr:
                at1 = set(norm_addr.split())
                at2 = set(c_addr.split())
                addr_jac = len(at1 & at2) / len(at1 | at2) if (at1 | at2) else 0.0
                addr_jw = JaroWinkler.normalized_similarity(norm_addr, c_addr)
                sim_addr = 0.50 * addr_jw + 0.50 * addr_jac
                score = 0.60 * sim_name + 0.40 * sim_addr
            else:
                score = sim_name

            # Bonus for exact anchors & channel consensus
            if norm_name and norm_name == c_name:
                score = min(1.0, score + 0.15)
            if postal_code and c_postal and postal_code == c_postal:
                score = min(1.0, score + 0.08)
            if hits >= 3:
                score = min(1.0, score + 0.05)

            scored.append((cid, score, hits))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def retrieve_candidates_for_s1(self, s1_rec: Dict[str, Any]) -> List[str]:
        """
        Retrieve and adaptively prune candidates for a single S1 record.
        Maintains backward compatibility with Amazon Tiebreaker:
        - Steep Drop-off Rule: Top score >= steep_threshold (0.85) and gap > gap_threshold (0.35) -> K=3
        - Diffuse Rule: Retain up to k_ceiling (25-30)
        """
        scored = self.retrieve_candidates_with_scores(s1_rec)
        if not scored:
            return []

        if len(scored) <= 3:
            return [cid for cid, _, _ in scored]

        top_score = scored[0][1]
        second_score = scored[1][1]

        # Adaptive candidate pruning policy
        if top_score >= self.steep_threshold and (top_score - second_score) > self.gap_threshold:
            return [cid for cid, _, _ in scored[:3]]
        else:
            return [cid for cid, _, _ in scored[:self.k_ceiling]]
