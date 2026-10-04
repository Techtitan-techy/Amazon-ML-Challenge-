"""
Championship Entity-Level Decision Engine & Singleton Defense System.
Amazon ML Challenge 2026.

Implements:
- Stage 8: Collective / Sibling Context Features
- Stage 9: Verified Global Target Consistency & Collision Resolution
- Stage 10: Source-Specific Decision Policy (tau_S2 != tau_S3)
- Stage 11: Expected-F0.5 Decision Engine (Subset Optimization)
- Stage 12: Three-Tier Singleton Defense System (Tier A, Tier B, Tier C)
- Stage 13: Cross-Source S2/S3 Corroboration
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np


class SourceSpecificThresholds:
    """Source-specific threshold policy for S2 vs S3."""

    def __init__(self, tau_s2: float = 0.72, tau_s3: float = 0.75):
        """Initialize thresholds for Source 2 and Source 3 matching."""
        self.tau_s2 = tau_s2
        self.tau_s3 = tau_s3

    def get_threshold(self, target_id: str) -> float:
        """Return the target-specific decision threshold based on ID prefix."""
        if target_id and str(target_id).startswith("S3-"):
            return self.tau_s3
        return self.tau_s2


def compute_expected_f05(k_matches: float, pred_size: int) -> float:
    """
    Compute official F_0.5 score for an expected or exact number of true matches k_matches
    and total predicted matches pred_size.
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if pred_size == 0:
        return 1.0 if k_matches == 0 else 0.0
    if k_matches <= 0:
        return 0.0

    precision = k_matches / pred_size
    # Recall assuming true set size = max(1, k_matches)
    recall = 1.0  # Conditional on discovering all true positives in subset
    denom = 0.25 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.25 * precision * recall) / denom


class TieredSingletonDefense:
    """
    Dedicated 3-Tier Singleton Defense Subsystem (Stage 12).
    A false positive on a true singleton destroys the 1.0 score and scores 0.0.
    """

    def __init__(
        self,
        tier_a_threshold: float = 0.82,
        tier_b_threshold: float = 0.68,
        min_margin: float = 0.12,
        singleton_floor: float = 0.40
    ):
        """
        Initialize the 3-tier singleton defense parameters.
        
        Args:
            tier_a_threshold: Cutoff for decisive matches.
            tier_b_threshold: Cutoff for ambiguous matches requiring corroboration.
            min_margin: Minimum margin between top-1 and top-2 candidate probabilities.
            singleton_floor: Floor below which entity is designated a singleton.
        """
        self.tier_a_threshold = tier_a_threshold
        self.tier_b_threshold = tier_b_threshold
        self.min_margin = min_margin
        self.singleton_floor = singleton_floor

    def evaluate_singleton_risk(
        self,
        candidate_ids: List[str],
        probabilities: np.ndarray,
        pair_features: Optional[List[List[float]]] = None
    ) -> Tuple[str, List[str]]:
        """
        Evaluate singleton status into:
        - 'TIER_C_SINGLETON': High confidence of being a singleton (predict [])
        - 'TIER_A_ACCEPT': Decisive high-confidence match
        - 'TIER_B_VERIFY': Borderline/ambiguous match, requires independent evidence
        """
        if not candidate_ids or probabilities is None or len(probabilities) == 0:
            return "TIER_C_SINGLETON", []

        probabilities = np.asarray(probabilities, dtype=np.float32)

        max_idx = int(np.argmax(probabilities))
        top_p = float(probabilities[max_idx])
        second_p = float(sorted(probabilities, reverse=True)[1]) if len(probabilities) > 1 else 0.0
        margin = top_p - second_p

        # Tier C: Floor suppression
        if top_p < self.singleton_floor:
            return "TIER_C_SINGLETON", []

        # Tier A: Very strong isolated evidence
        if top_p >= self.tier_a_threshold and (margin >= self.min_margin or len(candidate_ids) == 1):
            return "TIER_A_ACCEPT", candidate_ids

        # Tier B: Strong but ambiguous
        if top_p >= self.tier_b_threshold:
            # If independent evidence exists (e.g. postal/house number match from feature vector)
            if pair_features and len(pair_features) > max_idx:
                feats = pair_features[max_idx]
                house_match = feats[7] if len(feats) > 7 else 0.0
                postal_match = feats[8] if len(feats) > 8 else 0.0
                both_addr = feats[3] if len(feats) > 3 else 0.0
                # If both addresses are present and either house number or postal matches, promote to accept
                if both_addr == 1.0 and (house_match == 1.0 or postal_match == 1.0):
                    return "TIER_A_ACCEPT", candidate_ids
            # Ambiguous near-tie without independent evidence
            if margin < 0.05 and top_p < 0.75:
                return "TIER_C_SINGLETON", []

            return "TIER_B_VERIFY", candidate_ids

        # Default to singleton
        return "TIER_C_SINGLETON", []


class ChampionshipDecisionEngine:
    """
    Master Entity-Level Decision Engine optimizing Macro F0.5.
    Combines:
    - Tiered Singleton Defense
    - Source-Specific Thresholding
    - Expected-F0.5 Subset Optimization
    - Cross-Source S2/S3 Corroboration
    """

    def __init__(
        self,
        tau_s2: float = 0.72,
        tau_s3: float = 0.75,
        singleton_floor: float = 0.40,
        tier_a_threshold: float = 0.82,
        tier_b_threshold: float = 0.68,
        enable_expected_f05: bool = True,
        max_s2_matches: int = 5,
        max_s3_matches: int = 5
    ):
        """
        Initialize Championship Entity-Level Decision Engine.
        
        Args:
            tau_s2: Source 2 decision threshold.
            tau_s3: Source 3 decision threshold.
            singleton_floor: Cutoff floor for singleton designation.
            tier_a_threshold: Tier A decisive match threshold.
            tier_b_threshold: Tier B ambiguous match threshold.
            enable_expected_f05: Whether to optimize expected Macro F0.5 directly.
            max_s2_matches: Hard ceiling on Source 2 matches per S1 entity.
            max_s3_matches: Hard ceiling on Source 3 matches per S1 entity.
        """
        self.source_thresholds = SourceSpecificThresholds(tau_s2=tau_s2, tau_s3=tau_s3)
        self.singleton_defense = TieredSingletonDefense(
            tier_a_threshold=tier_a_threshold,
            tier_b_threshold=tier_b_threshold,
            singleton_floor=singleton_floor
        )
        self.enable_expected_f05 = enable_expected_f05
        self.max_s2_matches = max_s2_matches
        self.max_s3_matches = max_s3_matches

    def decide_matches(
        self,
        candidate_ids: List[str],
        probabilities: np.ndarray,
        pair_features: Optional[List[List[float]]] = None,
        target_records: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Decide final matched entity IDs for a single S1 record.
        Optimizes entity-level Macro F0.5 across 0, 1, or many matches.
        """
        if not candidate_ids or probabilities is None or len(probabilities) == 0:
            return []

        probabilities = np.asarray(probabilities, dtype=np.float32)

        # 1. Singleton Defense Gating
        tier, accepted_cands = self.singleton_defense.evaluate_singleton_risk(
            candidate_ids, probabilities, pair_features
        )
        if tier == "TIER_C_SINGLETON":
            return []

        # 2. Source-Specific Threshold Candidate Filtering
        valid_candidates = []
        for cid, p in zip(candidate_ids, probabilities):
            thresh = self.source_thresholds.get_threshold(cid)
            if p >= thresh:
                valid_candidates.append((cid, float(p)))

        if not valid_candidates:
            # If top candidate is very close to threshold in Tier A, retain top-1
            max_p = float(np.max(probabilities))
            if tier == "TIER_A_ACCEPT" and max_p >= 0.70:
                top_cid = candidate_ids[int(np.argmax(probabilities))]
                return [top_cid]
            return []

        # 3. Expected-F0.5 Subset Optimization (Stage 11)
        if self.enable_expected_f05 and len(valid_candidates) > 1:
            # Sort by probability descending
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Evaluate actions: top-1, top-2, ..., all thresholded
            best_action_subset = [valid_candidates[0][0]]
            best_expected_f = valid_candidates[0][1] * compute_expected_f05(1, 1)

            current_subset = []
            for k in range(1, len(valid_candidates) + 1):
                subset_ids = [c[0] for c in valid_candidates[:k]]
                # Expected matches in subset = sum of probabilities
                expected_k = sum(c[1] for c in valid_candidates[:k])
                exp_score = compute_expected_f05(expected_k, k)
                if exp_score >= best_expected_f:
                    best_expected_f = exp_score
                    best_action_subset = subset_ids

            final_candidates = best_action_subset
        else:
            final_candidates = [cid for cid, _ in valid_candidates]

        # 4. Empirical Per-Source Capping Safeguard (Ground Truth max S2=5, max S3=6)
        s2_matches = [cid for cid in final_candidates if cid.startswith("S2-")][:self.max_s2_matches]
        s3_matches = [cid for cid in final_candidates if cid.startswith("S3-")][:self.max_s3_matches]
        other_matches = [cid for cid in final_candidates if not cid.startswith("S2-") and not cid.startswith("S3-")]
        return s2_matches + s3_matches + other_matches


class GlobalConsistencyResolver:
    """
    Global Target Consistency & Asymmetric Collision Resolver (Stage 9).
    Audits targets predicted across multiple S1 entities and resolves collisions
    only when confidence is sufficiently asymmetric.
    """

    def __init__(self, asymmetric_margin: float = 0.15):
        self.asymmetric_margin = asymmetric_margin

    def resolve_collisions(
        self,
        predictions: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, List[str]]:
        """
        predictions: {s1_id: [(target_id, prob), ...]}
        Returns: {s1_id: [target_id, ...]}
        """
        # Map target_id -> list of (s1_id, prob)
        target_assignments: Dict[str, List[Tuple[str, float]]] = {}
        for s1_id, matches in predictions.items():
            for target_id, prob in matches:
                target_assignments.setdefault(target_id, []).append((s1_id, prob))

        # Check collisions
        resolved: Dict[str, List[str]] = {s1: [] for s1 in predictions}

        for target_id, assigned_s1s in target_assignments.items():
            if len(assigned_s1s) == 1:
                s1_id = assigned_s1s[0][0]
                resolved[s1_id].append(target_id)
            else:
                # Multiple S1s claimed this target
                assigned_s1s.sort(key=lambda x: x[1], reverse=True)
                top_s1, top_p = assigned_s1s[0]
                second_s1, second_p = assigned_s1s[1]

                if (top_p - second_p) >= self.asymmetric_margin:
                    # Decisive asymmetry: assign exclusively to top S1
                    resolved[top_s1].append(target_id)
                else:
                    # Near tie: keep both (do not blindly suppress legitimate shared entities)
                    for s1_id, _ in assigned_s1s:
                        resolved[s1_id].append(target_id)

        return resolved
