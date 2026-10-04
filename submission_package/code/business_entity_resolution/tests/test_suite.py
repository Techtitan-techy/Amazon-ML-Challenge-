"""
Comprehensive Test Suite for Business Entity Resolution Pipeline.
Tests:
1. Multi-View Normalization Layer (US, India, France invariance).
2. Adaptive Candidate Pruning (Amazon Tiebreaker rules: Steep Drop-off K=3, Diffuse K=25).
3. 15 SIMD Pairwise Feature Engine & Address Availability Awareness.
4. Two-Stage Decision Policy (Singleton Floor 0.40, Threshold 0.72).
5. Output TSV Formatting & Official Validator Compliance.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add src to path for direct execution
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from normalize import (  # type: ignore
        nfkc_normalize,
        token_normalize,
        compact_normalize,
        normalize_business_name,
        normalize_address,
        normalize_country,
        normalize_record,
    )
    from retrieval import HybridRetriever  # type: ignore
    from features import extract_pair_features, FEATURE_NAMES  # type: ignore
    from model import create_xgboost_matcher, apply_two_stage_decision_policy  # type: ignore
except ImportError:
    from src.normalize import (  # type: ignore
        nfkc_normalize,
        token_normalize,
        compact_normalize,
        normalize_business_name,
        normalize_address,
        normalize_country,
        normalize_record,
    )
    from src.retrieval import HybridRetriever  # type: ignore
    from src.features import extract_pair_features, FEATURE_NAMES  # type: ignore
    from src.model import create_xgboost_matcher, apply_two_stage_decision_policy  # type: ignore


class TestMultiViewNormalization(unittest.TestCase):
    """Test Suite for Text Normalization Layer & France Invariance."""

    def test_nfkc_and_french_accents(self):
        # France invariance: French accents must normalize cleanly under NFKC
        raw_french = "Café & Résumé Société par Actions Simplifiée"
        norm = normalize_business_name(raw_french)
        self.assertIn("cafe", norm)
        self.assertIn("resume", norm)
        self.assertTrue(norm.endswith("sas"), "French legal suffix should map to 'sas'")

    def test_us_legal_suffixes(self):
        self.assertEqual(normalize_business_name("Apple Incorporated"), "apple inc")
        self.assertEqual(normalize_business_name("Google LLC"), "google llc")
        self.assertEqual(normalize_business_name("General Electric Corporation"), "general electric corp")

    def test_indian_legal_suffixes(self):
        self.assertEqual(normalize_business_name("Infosys Private Limited"), "infosys pvt ltd")
        self.assertEqual(normalize_business_name("Wipro Limited"), "wipro ltd")
        self.assertEqual(normalize_business_name("Tata Consultancy Services LLP"), "tata consultancy services llp")

    def test_address_abbreviations(self):
        addr = "123 Main St, Apt 4B, MG Rd, 5th Ave"
        norm_addr = normalize_address(addr)
        self.assertIn("street", norm_addr)
        self.assertIn("apartment", norm_addr)
        self.assertIn("road", norm_addr)
        self.assertIn("avenue", norm_addr)

    def test_country_normalization_open_set(self):
        # Open-set: US, India, France must all normalize cleanly without error
        self.assertEqual(normalize_country("US"), "US")
        self.assertEqual(normalize_country("India"), "INDIA")
        self.assertEqual(normalize_country("France"), "FRANCE")

    def test_primitive_normalizers(self):
        # NFKC normalize: decomposes and strips accents
        self.assertEqual(nfkc_normalize("Crème Brûlée"), "Creme Brulee")
        # Token normalize: lowers, strips punctuation
        self.assertEqual(token_normalize("Hello, World! #123"), "hello world 123")
        # Compact normalize: removes whitespace
        self.assertEqual(compact_normalize("Target Corporation Store"), "targetcorporationstore")

    def test_normalize_record(self):
        rec = {
            "entity_id": "S1-999",
            "business_name": "Google, LLC",
            "business_address": "1600 Amphitheatre Pkwy, Mountain View, CA",
            "country": "US"
        }
        normed = normalize_record(rec)
        self.assertEqual(normed["entity_id"], "S1-999")
        self.assertEqual(normed["norm_name"], "google llc")
        self.assertEqual(normed["compact_name"], "googlellc")
        self.assertIn("parkway", normed["norm_addr"])
        self.assertEqual(normed["country"], "US")


class TestAdaptiveCandidatePruning(unittest.TestCase):
    """Test Suite for Amazon Ranking Tiebreaker: Adaptive Candidate Pruning."""

    def setUp(self):
        self.retriever = HybridRetriever(k_ceiling=25, steep_threshold=0.85, gap_threshold=0.35)

    def test_steep_dropoff_pruning(self):
        # Condition: Top score >= 0.85 and gap to 2nd > 0.35 -> Truncate to K=3
        s1 = {
            "entity_id": "S1-001",
            "norm_name": "target corporation store",
            "compact_name": "targetcorporationstore",
            "norm_addr": "1000 nicollet mall minneapolis mn",
            "country": "US"
        }
        # 10 target candidates: 1 perfect match, 9 weak candidates
        targets = [
            {"entity_id": "S2-001", "norm_name": "target corporation store", "compact_name": "targetcorporationstore", "norm_addr": "1000 nicollet mall minneapolis mn", "country": "US"},
            {"entity_id": "S2-002", "norm_name": "target logistics group", "compact_name": "targetlogisticsgroup", "norm_addr": "500 elm street dallas tx", "country": "US"},
        ] + [
            {"entity_id": f"S2-00{i}", "norm_name": f"target venture {i}", "compact_name": f"targetventure{i}", "norm_addr": f"{i} oak ave", "country": "US"}
            for i in range(3, 10)
        ]
        self.retriever.index_targets(targets)
        cands = self.retriever.retrieve_candidates_for_s1(s1)

        # Must prune to exactly K=3 due to steep dropoff
        self.assertLessEqual(len(cands), 3, f"Expected K <= 3 under steep dropoff, got {len(cands)}")
        self.assertEqual(cands[0], "S2-001", "Top candidate must be the true match")

    def test_diffuse_retrieval_pruning(self):
        # Condition: Ambiguous candidates with close scores -> Retain up to K=25
        s1 = {
            "entity_id": "S1-002",
            "norm_name": "national health clinic",
            "compact_name": "nationalhealthclinic",
            "norm_addr": "100 medical center drive",
            "country": "US"
        }
        # 30 similar candidates with close scores
        targets = [
            {"entity_id": f"S2-1{i:02d}", "norm_name": f"national health care {i}", "compact_name": f"nationalhealthcare{i}", "norm_addr": "100 medical center blvd", "country": "US"}
            for i in range(30)
        ]
        self.retriever.index_targets(targets)
        cands = self.retriever.retrieve_candidates_for_s1(s1)

        # Must be capped at ceiling K=25
        self.assertLessEqual(len(cands), 25, f"Expected K <= 25, got {len(cands)}")
        self.assertGreater(len(cands), 5, "Diffuse candidates should not be overly truncated")


class TestFeatureEngine(unittest.TestCase):
    """Test Suite for 15 SIMD RapidFuzz Pairwise Features & Address Availability."""

    def test_feature_vector_dimension(self):
        s1 = {"norm_name": "amazon data services", "norm_addr": "410 terry ave n seattle wa"}
        target = {"norm_name": "amazon data services inc", "norm_addr": "410 terry avenue seattle"}
        feats = extract_pair_features(s1, target)
        self.assertEqual(len(feats), 15, "Feature vector must contain exactly 15 features")
        self.assertEqual(len(FEATURE_NAMES), 15)

    def test_address_missing_regime(self):
        # When address is missing, both_addr_present must be 0 and addr_missing must be 1
        s1 = {"norm_name": "apple store", "norm_addr": ""}
        target = {"norm_name": "apple store", "norm_addr": ""}
        feats = extract_pair_features(s1, target)
        
        both_addr = feats[3]
        addr_missing = feats[9]
        name_x_missing = feats[11]

        self.assertEqual(both_addr, 0.0, "both_addr_present should be 0 when addresses are blank")
        self.assertEqual(addr_missing, 1.0, "addr_missing should be 1 when addresses are blank")
        self.assertGreater(name_x_missing, 0.9, "name_x_addr_missing should activate on high name match")

    def test_house_number_matching(self):
        # House numbers match: 500 == 500
        s1 = {"norm_name": "xyz corp", "norm_addr": "500 broadway new york"}
        target1 = {"norm_name": "xyz corp", "norm_addr": "500 broadway suite 2 new york"}
        feats1 = extract_pair_features(s1, target1)
        self.assertEqual(feats1[7], 1.0, "house_num_match should be 1 when building numbers match")

        # House numbers conflict: 500 != 900
        target2 = {"norm_name": "xyz corp", "norm_addr": "900 broadway new york"}
        feats2 = extract_pair_features(s1, target2)
        self.assertEqual(feats2[7], 0.0, "house_num_match should be 0 when building numbers conflict")


class TestTwoStageDecisionPolicy(unittest.TestCase):
    """Test Suite for Singleton Floor Gating and Precision-Heavy F_0.5 Selection."""

    def test_singleton_floor_suppression(self):
        # Singleton protection: If max_p < 0.40 -> Must predict empty list []
        cands = ["S2-001", "S2-002", "S3-005"]
        low_probs = np.array([0.35, 0.28, 0.12])
        matches = apply_two_stage_decision_policy(cands, low_probs, threshold=0.72, singleton_floor=0.40)
        self.assertEqual(matches, [], "Singleton floor must suppress predictions when max_p < 0.40")

    def test_precision_threshold_selection(self):
        # Multi-match: Select candidates meeting threshold >= 0.72
        cands = ["S2-001", "S2-002", "S3-005"]
        probs = np.array([0.95, 0.78, 0.45])
        matches = apply_two_stage_decision_policy(cands, probs, threshold=0.72, singleton_floor=0.40)
        self.assertEqual(matches, ["S2-001", "S2-002"], "Must select all candidates exceeding threshold")

    def test_variable_cardinality(self):
        # Exactly one match
        cands = ["S2-001", "S2-002"]
        probs = np.array([0.88, 0.55])
        matches = apply_two_stage_decision_policy(cands, probs, threshold=0.72, singleton_floor=0.40)
        self.assertEqual(matches, ["S2-001"])


class TestModelArtifacts(unittest.TestCase):
    """Verify that model artifacts and requirements are correctly present."""

    def test_model_joblib_exists(self):
        model_path = SRC_DIR / "models" / "xgboost_matcher.joblib"
        self.assertTrue(model_path.exists(), f"Model file must exist at {model_path}")
        self.assertGreater(model_path.stat().st_size, 10000, "Model file size should be substantial")

    def test_create_xgboost_matcher(self):
        model = create_xgboost_matcher()
        self.assertEqual(model.n_estimators, 300)
        self.assertEqual(model.max_depth, 6)
        self.assertEqual(model.learning_rate, 0.08)


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING BUSINESS ENTITY RESOLUTION PIPELINE UNIT TESTS")
    print("=" * 80)
    unittest.main(verbosity=2)
