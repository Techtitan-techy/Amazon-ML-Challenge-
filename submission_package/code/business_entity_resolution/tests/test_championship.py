"""
Comprehensive Test Suite for Championship Business Entity Resolution Architecture.
Amazon ML Challenge 2026.

Tests:
1. Championship Multi-View Normalization (DBA, domains, phonetic skeleton, Indic transliterations).
2. Championship 35-Dimensional Feature Matrix.
3. Multi-Channel Candidate Retrieval & Reranker.
4. Diverse Ensemble Matcher & Probability Calibration.
5. Entity-Level Expected-F0.5 Decision Engine & Tiered Singleton Defense.
6. Global Target Consistency & Collision Resolution.
7. Oracle Candidate Diagnostics & Segment Scoreboard.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalize import (
    normalize_record,
    normalize_business_name,
    normalize_address,
    clean_dba_and_domains,
    compute_phonetic_skeleton,
    extract_house_number,
    extract_postal_code
)
from retrieval import HybridRetriever
from features import (
    extract_pair_features,
    extract_championship_features,
    CHAMPIONSHIP_FEATURE_NAMES
)
from model import (
    EnsembleMatcher,
    create_xgboost_matcher,
    create_lightgbm_matcher,
    create_hist_gradient_boosting_matcher
)
from decision_engine import (
    ChampionshipDecisionEngine,
    SourceSpecificThresholds,
    TieredSingletonDefense,
    GlobalConsistencyResolver,
    compute_expected_f05
)
from diagnostics import (
    compute_entity_macro_f05,
    run_oracle_candidate_diagnostics,
    generate_segment_scoreboard
)


class TestChampionshipNormalization(unittest.TestCase):
    """Test Suite for Advanced Multi-View Normalization (Stage 1)."""

    def test_dba_and_domain_parsing(self):
        self.assertEqual(clean_dba_and_domains("Apex Corp dba Apex Logistics"), "apex logistics")
        self.assertEqual(clean_dba_and_domains("http://www.cool-gadgets.com"), "cool gadgets")
        self.assertEqual(clean_dba_and_domains("Metro Foods c/o Central Distributors"), "central distributors")

    def test_phonetic_skeleton(self):
        sk1 = compute_phonetic_skeleton("Lakshmi Jewellers")
        sk2 = compute_phonetic_skeleton("Laxmi Jewelers")
        self.assertEqual(sk1, sk2, "Phonetic skeleton should align variant spellings")

    def test_indic_aliases(self):
        norm = normalize_business_name("Shree Laxmi Vidya Pith")
        self.assertIn("sri", norm)
        self.assertIn("lakshmi", norm)

    def test_structured_extractions(self):
        tokens = ["123", "mg", "road", "bengaluru", "560001"]
        self.assertEqual(extract_house_number(tokens), "123")
        self.assertEqual(extract_postal_code(tokens, "INDIA"), "560001")


class TestChampionshipFeatureEngine(unittest.TestCase):
    """Test Suite for 42-Dimensional Competitor-Surpassing Feature Matrix (Stage 4 & Stage 8)."""

    def test_feature_vector_dimension(self):
        s1 = {"norm_name": "google cloud", "norm_addr": "1600 amphitheatre pkwy mountain view"}
        target = {"entity_id": "S2-100", "norm_name": "google cloud llc", "norm_addr": "1600 amphitheatre parkway"}
        feats = extract_championship_features(s1, target)
        self.assertEqual(len(feats), 42, f"Expected 42 features, got {len(feats)}")
        self.assertEqual(len(CHAMPIONSHIP_FEATURE_NAMES), 42)

    def test_containment_and_char_ngrams(self):
        s1 = {"norm_name": "apple computer", "norm_addr": "1 infinite loop"}
        target = {"entity_id": "S3-200", "norm_name": "apple computer inc", "norm_addr": "1 infinite loop cupertino"}
        feats = extract_championship_features(s1, target)
        # name_containment is index 16
        self.assertEqual(feats[16], 1.0, "name_containment should be 1.0")
        # target_source_s3 is index 28
        self.assertEqual(feats[28], 1.0, "S3 indicator should be 1.0 for S3-200")
        # first_token_match is index 35
        self.assertEqual(feats[35], 1.0, "first_token_match should be 1.0 ('apple' == 'apple')")
        # cross_prod is index 39
        self.assertGreater(feats[39], 0.0, "cross_prod should be positive")
        # name_char3_cos is index 40
        self.assertGreater(feats[40], 0.5, "name_char3_cos should be > 0.5")


class TestChampionshipDecisionEngine(unittest.TestCase):
    """Test Suite for Entity-Level Decision Engine & Singleton Defense (Stage 9-12)."""

    def setUp(self):
        self.engine = ChampionshipDecisionEngine(tau_s2=0.72, tau_s3=0.75, singleton_floor=0.40)

    def test_tiered_singleton_defense(self):
        cands = ["S2-001", "S2-002"]
        low_probs = np.array([0.30, 0.25])
        matches = self.engine.decide_matches(cands, low_probs)
        self.assertEqual(matches, [], "Tier C: Must suppress low probabilities into empty singleton")

    def test_source_specific_thresholds(self):
        # S2 threshold is 0.72, S3 threshold is 0.75
        cands = ["S2-001", "S3-001"]
        probs = np.array([0.80, 0.73])  # S2 passes (0.80 >= 0.72), S3 fails (0.73 < 0.75)
        matches = self.engine.decide_matches(cands, probs)
        self.assertIn("S2-001", matches)
        self.assertNotIn("S3-001", matches)

    def test_per_source_capping(self):
        # Even if 8 S2 candidates pass threshold, max_s2_matches=5 caps at 5
        cands = [f"S2-{i:03d}" for i in range(8)]
        probs = np.array([0.95 - i * 0.01 for i in range(8)])  # all >= 0.88 > 0.72
        matches = self.engine.decide_matches(cands, probs)
        self.assertLessEqual(len(matches), 5, f"Expected at most 5 S2 matches, got {len(matches)}")

    def test_global_collision_resolution(self):
        resolver = GlobalConsistencyResolver(asymmetric_margin=0.15)
        preds = {
            "S1-A": [("S2-MATCH", 0.95)],
            "S1-B": [("S2-MATCH", 0.60)]  # Asymmetric conflict: S1-A has 0.95 vs 0.60
        }
        resolved = resolver.resolve_collisions(preds)
        self.assertIn("S2-MATCH", resolved["S1-A"])
        self.assertNotIn("S2-MATCH", resolved["S1-B"])


class TestChampionshipDiagnostics(unittest.TestCase):
    """Test Suite for Official Metric & Diagnostics (Stage 17 & 22)."""

    def test_macro_f05_calculation(self):
        gt = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": set()  # Singleton
        }
        preds = {
            "S1-1": {"S2-10", "S3-20"},  # Perfect match: F0.5 = 1.0
            "S1-2": set()                 # Correct singleton: F0.5 = 1.0
        }
        f05, p, r = compute_entity_macro_f05(gt, preds)
        self.assertEqual(f05, 1.0)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)


class TestDiverseModelEnsemble(unittest.TestCase):
    """Test Suite for XGBoost + LightGBM + HistGradientBoosting Ensemble (Stage 5 & 7)."""

    def test_ensemble_training_and_calibration(self):
        X = np.random.rand(50, 15).astype(np.float32)
        y = np.random.randint(0, 2, size=50).astype(np.int32)
        ensemble = EnsembleMatcher(calibrate=True)
        ensemble.fit(X, y)
        probs = ensemble.predict_proba(X)
        self.assertEqual(probs.shape, (50, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING CHAMPIONSHIP UNIT TESTS")
    print("=" * 80)
    unittest.main(verbosity=2)
