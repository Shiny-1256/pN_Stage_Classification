"""
Multimodal Machine Learning Classifier for Branch 3.
Trains an ensemble tree model (Random Forest / Gradient Boosting / XGBoost)
on joint PathDL + ClinicalML features, computes calibrated 5-class pN-stage posteriors,
and ranks feature importances.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class MultimodalStagingClassifier:
    """
    Predicts 5-class patient pN stage from combined PathDL + ClinicalML feature vectors.
    """

    STAGE_NAMES = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]
    STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_NAMES)}

    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.observed_classes_: np.ndarray = np.array([])

        if model_type == "xgboost" and HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                eval_metric="mlogloss",
                random_state=random_state,
                subsample=1.0,
                colsample_bytree=1.0,
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                random_state=random_state,
            )
        else:
            # Default Random Forest ensemble: robust, non-overfitting on small cohorts
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                random_state=random_state,
            )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fits the scaler and multi-class model on the feature matrix.
        """
        if feature_names:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        unique_classes, y_encoded = np.unique(y, return_inverse=True)
        self.observed_classes_ = unique_classes

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_encoded)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes calibrated 5-class posterior probability matrix [N, 5].
        """
        X_scaled = self.scaler.transform(X)
        raw_probs = self.model.predict_proba(X_scaled)

        N = X.shape[0]
        # Base prior for unobserved classes
        prior_weight = 0.02
        full_probs = np.full((N, 5), prior_weight, dtype=np.float32)

        for local_idx, orig_cls in enumerate(self.observed_classes_):
            if 0 <= orig_cls < 5:
                full_probs[:, orig_cls] += raw_probs[:, local_idx]

        # Row-normalize
        row_sums = full_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        full_probs = full_probs / row_sums

        return full_probs

    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predicts the argmax pN stage name for each sample.
        """
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return [self.STAGE_NAMES[idx] for idx in pred_indices]

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Returns normalized feature importance dictionary.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            total = sum(importances) or 1.0
            norm_imp = [float(imp / total) for imp in importances]
            return dict(sorted(zip(self.feature_names, norm_imp), key=lambda x: x[1], reverse=True))
        return {f: 1.0 / max(1, len(self.feature_names)) for f in self.feature_names}

    def save(self, filepath: str | Path):
        """
        Serializes model checkpoint to disk.
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "observed_classes_": self.observed_classes_,
            "model_type": self.model_type,
            "random_state": self.random_state,
        }
        joblib.dump(payload, p)
        print(f"Saved Branch 3 model to: {p}")

    @classmethod
    def load(cls, filepath: str | Path) -> "MultimodalStagingClassifier":
        """
        Loads model checkpoint from disk.
        """
        payload = joblib.load(filepath)
        instance = cls(model_type=payload.get("model_type", "random_forest"), random_state=payload.get("random_state", 42))
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.feature_names = payload["feature_names"]
        instance.observed_classes_ = payload.get("observed_classes_", np.array([]))
        return instance
