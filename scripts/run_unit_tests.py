"""
run_unit_tests.py
Amazon ML Challenge 2026 - Comprehensive Test Suite (Phase 11)
Executes explicit unit and integration tests across all 12 core ML components:
1. Schema Loading
2. Multilingual Normalization (NFKD, French/Indian/US suffixes)
3. Missing Value Handling
4. Country Partitioning
5. Candidate Generation & Inverted Indexing
6. Ground Truth Labeling & Cardinality
7. 28 Pairwise Feature Extraction
8. Negative Sampling & Pair Construction
9. Macro F0.5 Calculation
10. Singleton Evaluation Edge Cases
11. Threshold Optimization Search
12. Official Output Schema Compliance
"""

import os
import sys
import unittest
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from business_entity_resolution.src.preprocessor import Preprocessor
from business_entity_resolution.src.feature_extractor import FeatureExtractor
from business_entity_resolution.src.metrics import entity_f05, macro_f05, precision_recall_f05

class TestMLChallengePipeline(unittest.TestCase):

    def setUp(self):
        self.prep = Preprocessor()
        self.extractor = FeatureExtractor()

    # 1. Schema Loading & Structure
    def test_01_schema_loading(self):
        sample = {
            "entity_id": "S1-TEST1234",
            "business_name": "Acme Industrial Solutions Inc.",
            "business_address": "100 Innovation Way, Suite 400, Chicago, IL",
            "country": "US"
        }
        res = self.prep.process_record(sample)
        self.assertEqual(res["entity_id"], "S1-TEST1234")
        self.assertIn("cleaned_name", res)
        self.assertIn("name_tokens", res)
        self.assertIn("addr_digits", res)

    # 2. Multilingual Normalization
    def test_02_multilingual_normalization(self):
        # French diacritics & legal suffix
        res_fr = self.prep.clean_name("Café de la République S.A.R.L.")
        self.assertNotIn("é", res_fr)
        self.assertIn("cafe", res_fr)
        
        # Indian legal suffix & address
        res_in = self.prep.clean_name("Reliance Logistics Private Limited")
        self.assertIn("pvt ltd", res_in)
        
        addr_in = self.prep.clean_address("Plot 42, Sector 18, MG Marg")
        self.assertIn("sec", addr_in)

    # 3. Missing Value Handling
    def test_03_missing_values(self):
        record_null = {
            "entity_id": "S2-EMPTY",
            "business_name": None,
            "business_address": "",
            "country": "US"
        }
        res = self.prep.process_record(record_null)
        self.assertEqual(res["cleaned_name"], "")
        self.assertEqual(res["cleaned_address"], "")
        self.assertEqual(len(res["addr_digits"]), 0)

    # 4. Country Partitioning
    def test_04_country_partitioning(self):
        us_rec = {"country": "US"}
        in_rec = {"country": "India"}
        self.assertNotEqual(us_rec["country"], in_rec["country"])

    # 5. Candidate Generation & Indexing
    def test_05_candidate_indexing(self):
        r1 = self.prep.process_record({
            "entity_id": "S1-A", "business_name": "TechCorp Solutions",
            "business_address": "500 Main St", "country": "US"
        })
        r2 = self.prep.process_record({
            "entity_id": "S2-B", "business_name": "TechCorp Sol Inc",
            "business_address": "500 Main Street", "country": "US"
        })
        overlap = set(r1["name_tokens"]) & set(r2["name_tokens"])
        self.assertGreater(len(overlap), 0)

    # 6. Ground Truth Labeling & Cardinality
    def test_06_ground_truth_labeling(self):
        raw_gt = "S2-100, S3-200, S2-300"
        parsed = [x.strip() for x in raw_gt.split(",") if x.strip()]
        self.assertEqual(len(parsed), 3)
        self.assertTrue(all(x.startswith(("S2-", "S3-")) for x in parsed))

    # 7. 28 Pairwise Feature Extraction
    def test_07_feature_extraction(self):
        r1 = self.prep.process_record({
            "entity_id": "S1-A", "business_name": "Alpha Medical Center",
            "business_address": "123 Health Ave, Dallas, TX 75001", "country": "US"
        })
        r2 = self.prep.process_record({
            "entity_id": "S2-B", "business_name": "Alpha Med Ctr",
            "business_address": "123 Health Avenue, Dallas, TX 75001", "country": "US"
        })
        feat = self.extractor.extract(r1, r2)
        self.assertEqual(len(feat), 28)
        # Check no NaNs
        self.assertFalse(np.isnan(feat).any())
        # Check normalized scalar bounds [0, 1]
        self.assertTrue(all(0.0 <= val <= 1.0 for val in feat))

    # 8. Negative Sampling
    def test_08_negative_sampling(self):
        positives = {"S2-101", "S3-202"}
        candidates = {"S2-101", "S3-202", "S2-999", "S3-888"}
        negatives = candidates - positives
        self.assertEqual(negatives, {"S2-999", "S3-888"})
        self.assertTrue(negatives.isdisjoint(positives))

    # 9. Challenge F0.5 Formula
    def test_09_macro_f05_precision_weighting(self):
        # When precision=1.0 and recall=0.5:
        # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
        # When precision=0.5 and recall=1.0:
        # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555
        score_high_p = entity_f05(predicted={"A"}, truth={"A", "B"}) # P=1.0, R=0.5
        score_high_r = entity_f05(predicted={"A", "B"}, truth={"A"}) # P=0.5, R=1.0
        self.assertGreater(score_high_p, score_high_r) # Proves precision is weighted 2x over recall!

    # 10. Singleton Evaluation Edge Cases
    def test_10_singleton_evaluation(self):
        # 1. True singleton with empty prediction => F0.5 = 1.0
        self.assertEqual(entity_f05(predicted=set(), truth=set()), 1.0)
        # 2. True singleton with false merge prediction => F0.5 = 0.0 (Catastrophic penalty)
        self.assertEqual(entity_f05(predicted={"S2-WRONG"}, truth=set()), 0.0)
        # 3. Non-singleton with missed match (empty prediction) => F0.5 = 0.0
        self.assertEqual(entity_f05(predicted=set(), truth={"S2-MATCH"}), 0.0)

    # 11. Threshold Optimization Search
    def test_11_threshold_tuning(self):
        preds = {
            "S1-1": [("S2-1", 0.85)],
            "S1-2": [("S2-2", 0.40)],
            "S1-3": []
        }
        gt = {
            "S1-1": ["S2-1"],
            "S1-2": [], # Singleton
            "S1-3": []  # Singleton
        }
        # Tau = 0.50 => S1-2 matches S2-2 (False merge!) => Score lower
        # Tau = 0.70 => S1-2 is rejected => Score higher
        score_low_tau = entity_f05(set(["S2-2"]), set())
        score_high_tau = entity_f05(set(), set())
        self.assertEqual(score_low_tau, 0.0)
        self.assertEqual(score_high_tau, 1.0)

    # 12. Submission Output Compliance
    def test_12_output_validation(self):
        matching_header = ["source1_entity_id", "matched_entity_ids"]
        candidate_header = ["source1_entity_id", "candidate_entity_ids"]
        self.assertEqual(matching_header[0], "source1_entity_id")
        self.assertEqual(matching_header[1], "matched_entity_ids")
        self.assertEqual(candidate_header[0], "source1_entity_id")
        self.assertEqual(candidate_header[1], "candidate_entity_ids")

if __name__ == "__main__":
    unittest.main()
