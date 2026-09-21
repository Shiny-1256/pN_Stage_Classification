"""
Branch 3: Multimodal PathDL + ClinicalML Module
Integrates whole-slide quantitative lesion burden with clinical and biomarker profiles.
"""

from .pathology_features import PathologyFeatureExtractor
from .clinical_features import ClinicalFeatureManager
from .multimodal_dataset import MultimodalDatasetBuilder
from .multimodal_model import MultimodalStagingClassifier

__all__ = [
    "PathologyFeatureExtractor",
    "ClinicalFeatureManager",
    "MultimodalDatasetBuilder",
    "MultimodalStagingClassifier",
]
