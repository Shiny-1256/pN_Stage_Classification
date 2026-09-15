"""
Branch 1: Deep Learning Ensemble (ResNet + DenseNet)
Handles patch-level feature learning, slide-level pooling, and patient-level pN-stage prediction.
"""

from .patch_dataset import H5PatchDataset
from .resnet_model import ResNetPatchClassifier
from .densenet_model import DenseNetPatchClassifier
from .patient_aggregator import PatientSlideAggregator
from .stage_predictor import PatientStagePredictor
from .ensemble_module import Branch1Ensemble

__all__ = [
    "H5PatchDataset",
    "ResNetPatchClassifier",
    "DenseNetPatchClassifier",
    "PatientSlideAggregator",
    "PatientStagePredictor",
    "Branch1Ensemble",
]
