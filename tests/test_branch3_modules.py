"""
Unit Tests for Branch 3 (Multimodal Analysis) and SuperNet Late Fusion.
Verifies Pathology Feature Extraction, Clinical Tabular Encoding,
Multimodal Dataset Builder, Multimodal Classifier, and SuperNet Consensus Staging.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.branch3.pathology_features import PathologyFeatureExtractor
from src.branch3.clinical_features import ClinicalFeatureManager
from src.branch3.multimodal_dataset import MultimodalDatasetBuilder
from src.branch3.multimodal_model import MultimodalStagingClassifier
from src.fusion.late_fusion_model import SuperNetConsensusClassifier


class TestBranch3AndSuperNet(unittest.TestCase):

    def test_pathology_feature_extraction(self):
        extractor = PathologyFeatureExtractor()
        dummy_slides = [
            {"slide_id": "p1_n0", "annotated_tumor_burden_ratio": 0.05, "total_tissue_patches": 1000, "tumor_patches_annotated": 50},
            {"slide_id": "p1_n1", "annotated_tumor_burden_ratio": 0.0, "total_tissue_patches": 800, "tumor_patches_annotated": 0},
            {"slide_id": "p1_n2", "annotated_tumor_burden_ratio": 0.0, "total_tissue_patches": 900, "tumor_patches_annotated": 0},
            {"slide_id": "p1_n3", "annotated_tumor_burden_ratio": 0.0, "total_tissue_patches": 700, "tumor_patches_annotated": 0},
            {"slide_id": "p1_n4", "annotated_tumor_burden_ratio": 0.0, "total_tissue_patches": 600, "tumor_patches_annotated": 0},
        ]
        feats = extractor.extract_features_for_patient("patient_test", dummy_slides)
        self.assertIn("cumulative_tumor_burden", feats)
        self.assertAlmostEqual(feats["cumulative_tumor_burden"], 0.05, places=4)
        self.assertAlmostEqual(feats["max_nodal_burden"], 0.05, places=4)
        self.assertEqual(feats["pos_nodes_count"], 1.0)
        self.assertEqual(feats["total_tissue_patches"], 4000.0)

    def test_clinical_features(self):
        manager = ClinicalFeatureManager()
        prof = manager.get_patient_profile("patient_000")
        encoded = manager.encode_profile(prof)
        self.assertIn("clin_age", encoded)
        self.assertIn("clin_tumor_size_mm", encoded)
        self.assertIn("clin_t_stage_ordinal", encoded)
        self.assertEqual(encoded["clin_t_stage_ordinal"], 1.0)

    def test_multimodal_dataset_builder(self):
        summary_csv = PROJECT_ROOT / "output" / "branch3_multimodal" / "slides_multimodal_summary.csv"
        meta_json = PROJECT_ROOT / "output" / "slide_visualizations" / "slide_meta.json"
        stage_csv = PROJECT_ROOT.parent / "dataset" / "stage_labels.csv"

        builder = MultimodalDatasetBuilder(summary_csv=summary_csv, meta_json=meta_json, stage_csv=stage_csv)
        df = builder.build_dataset(["patient_000", "patient_001"])
        self.assertEqual(len(df), 2)
        X, y, feat_cols = builder.get_features_and_targets(df)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(len(y), 2)
        self.assertGreater(len(feat_cols), 10)

    def test_multimodal_classifier(self):
        clf = MultimodalStagingClassifier(model_type="random_forest", random_state=42)
        X = np.array([
            [0.0, 0.0, 0.0, 50.0, 15.0],
            [0.0, 0.0, 0.0, 60.0, 12.0],
            [0.001, 1.0, 0.0, 48.0, 22.0],
            [0.08, 2.0, 2.0, 58.0, 38.0],
        ], dtype=np.float32)
        y = np.array([0, 0, 1, 3], dtype=np.int64)

        clf.fit(X, y, feature_names=["b1", "b2", "b3", "c1", "c2"])
        probs = clf.predict_proba(X)
        self.assertEqual(probs.shape, (4, 5))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-4))

        preds = clf.predict(X)
        self.assertEqual(len(preds), 4)

        importances = clf.get_feature_importance()
        self.assertEqual(len(importances), 5)

    def test_supernet_late_fusion(self):
        supernet = SuperNetConsensusClassifier(
            weights={"branch1": 0.30, "branch2": 0.40, "branch3": 0.30}
        )

        p1 = {"pN0": 0.85, "pN0(i+)": 0.05, "pN1mi": 0.04, "pN1": 0.03, "pN2": 0.03}
        p2 = {"pN0": 0.90, "pN0(i+)": 0.03, "pN1mi": 0.03, "pN1": 0.02, "pN2": 0.02}
        p3 = {"pN0": 0.80, "pN0(i+)": 0.10, "pN1mi": 0.04, "pN1": 0.03, "pN2": 0.03}

        c_stage, c_conf, c_dist, audit = supernet.fuse_probabilities(p1, p2, p3)
        self.assertEqual(c_stage, "pN0")
        self.assertGreater(c_conf, 0.75)
        self.assertTrue(audit["all_branches_unanimous"])
        self.assertAlmostEqual(audit["agreement_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
