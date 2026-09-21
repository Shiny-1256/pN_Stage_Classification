"""
SuperNet Late Fusion Meta-Learner.
Ensembles predictions from:
- Branch 1: Cellular-Tissue Deep Learning Ensemble (ResNet-50 + DenseNet-121)
- Branch 2: Selective Neighborhood Attention (SNA Spatial Graph Model)
- Branch 3: Multimodal Analysis (PathDL WSI Burden + ClinicalML Profiles)
Produces the final patient-level consensus pN stage and calibrated posterior vector.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


class SuperNetConsensusClassifier:
    """
    Stacking meta-learner aggregating all 3 methodology branches into a consensus decision.
    """

    STAGE_NAMES = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]
    STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_NAMES)}

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_stacking: bool = True,
        random_state: int = 42,
    ):
        # Default prior weights if stacking not yet trained:
        # Branch 2 (graph lesion area) has highest structural precision for AJCC staging (0.45)
        # Branch 1 (cellular deep ensemble) contributes 0.30
        # Branch 3 (multimodal clinical + burden) contributes 0.25
        self.weights = weights or {"branch1": 0.30, "branch2": 0.45, "branch3": 0.25}
        self.use_stacking = use_stacking
        self.meta_classifier = LogisticRegression(C=1.0, max_iter=200, random_state=random_state)
        self.is_fitted = False

    def construct_meta_vector(
        self,
        p_b1: Dict[str, float] | List[float] | np.ndarray,
        p_b2: Dict[str, float] | List[float] | np.ndarray,
        p_b3: Dict[str, float] | List[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Flattens 5-class posteriors from the 3 branches into a 15-dimensional meta-feature vector.
        """
        def to_vec(p):
            if isinstance(p, dict):
                return np.array([float(p.get(s, 0.0)) for s in self.STAGE_NAMES], dtype=np.float32)
            return np.array(p, dtype=np.float32)

        v1 = to_vec(p_b1)
        v2 = to_vec(p_b2)
        v3 = to_vec(p_b3)

        return np.concatenate([v1, v2, v3], axis=-1)

    def fit_meta_learner(
        self,
        meta_features: np.ndarray,
        ground_truth_targets: np.ndarray,
    ):
        """
        Trains stacking logistic regression meta-model.
        """
        self.meta_classifier.fit(meta_features, ground_truth_targets)
        self.is_fitted = True
        return self

    def fuse_probabilities(
        self,
        p_b1: Dict[str, float] | List[float] | np.ndarray,
        p_b2: Dict[str, float] | List[float] | np.ndarray,
        p_b3: Dict[str, float] | List[float] | np.ndarray,
    ) -> Tuple[str, float, Dict[str, float], Dict[str, Any]]:
        """
        Fuses branch predictions into consensus stage, confidence, and distribution.
        """
        def to_vec(p):
            if isinstance(p, dict):
                return np.array([float(p.get(s, 0.0)) for s in self.STAGE_NAMES], dtype=np.float32)
            return np.array(p, dtype=np.float32)

        v1 = to_vec(p_b1)
        v2 = to_vec(p_b2)
        v3 = to_vec(p_b3)

        if self.is_fitted:
            meta_vec = self.construct_meta_vector(v1, v2, v3).reshape(1, -1)
            raw_probs = self.meta_classifier.predict_proba(meta_vec)[0]
            # Map observed classes to full 5 classes
            full_probs = np.zeros(5, dtype=np.float32)
            for idx, c in enumerate(self.meta_classifier.classes_):
                full_probs[c] = raw_probs[idx]
            fused_vec = full_probs / (full_probs.sum() or 1.0)
        else:
            # Weighted average blend using calibrated branch weights
            w1 = self.weights.get("branch1", 0.30)
            w2 = self.weights.get("branch2", 0.45)
            w3 = self.weights.get("branch3", 0.25)
            fused_vec = (w1 * v1) + (w2 * v2) + (w3 * v3)
            fused_vec = fused_vec / (fused_vec.sum() or 1.0)

        pred_idx = int(np.argmax(fused_vec))
        consensus_stage = self.STAGE_NAMES[pred_idx]
        confidence = float(fused_vec[pred_idx])
        distribution = {s: float(fused_vec[i]) for i, s in enumerate(self.STAGE_NAMES)}

        # Determine individual branch argmax predictions to calculate agreement
        pred_b1 = self.STAGE_NAMES[int(np.argmax(v1))]
        pred_b2 = self.STAGE_NAMES[int(np.argmax(v2))]
        pred_b3 = self.STAGE_NAMES[int(np.argmax(v3))]

        branch_preds = [pred_b1, pred_b2, pred_b3]
        agreement_count = branch_preds.count(consensus_stage)
        agreement_ratio = float(agreement_count / 3.0)

        audit_trail = {
            "branch1_pred": pred_b1,
            "branch2_pred": pred_b2,
            "branch3_pred": pred_b3,
            "consensus_stage": consensus_stage,
            "consensus_confidence": confidence,
            "agreement_ratio": agreement_ratio,
            "all_branches_unanimous": (pred_b1 == pred_b2 == pred_b3),
        }

        return consensus_stage, confidence, distribution, audit_trail

    def save(self, filepath: str | Path):
        """
        Saves SuperNet model to disk.
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        print(f"Saved SuperNet consensus model to: {p}")

    @classmethod
    def load(cls, filepath: str | Path) -> "SuperNetConsensusClassifier":
        """
        Loads SuperNet model from disk.
        """
        return joblib.load(filepath)
