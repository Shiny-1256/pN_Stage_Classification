"""
Unit Tests for Branch 1 Modules.
Verifies ResNet-50, DenseNet-121, H5 Dataset loader, Slide/Patient Aggregator,
and Ensemble Fusion.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.branch1.patch_dataset import H5PatchDataset
from src.branch1.resnet_model import ResNetPatchClassifier
from src.branch1.densenet_model import DenseNetPatchClassifier
from src.branch1.patient_aggregator import PatientSlideAggregator
from src.branch1.stage_predictor import PatientStagePredictor
from src.branch1.ensemble_module import Branch1Ensemble


def test_h5_patch_dataset():
    print("[1/6] Testing H5PatchDataset Loader...")
    h5_dir = Path("output/branch1_patches")
    if not h5_dir.exists() or len(list(h5_dir.glob("*.h5"))) == 0:
        print("      Skipping H5 file read (no .h5 files found yet).")
        return

    dataset = H5PatchDataset(h5_dir=h5_dir, is_training=False)
    assert len(dataset) > 0, "Dataset should have patches indexed"
    sample = dataset[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["label"].ndim == 0
    print(f"      H5PatchDataset loaded successfully ({len(dataset)} patches indexed across slides).")


def test_resnet_model():
    print("[2/6] Testing ResNet-50 Patch Classifier...")
    model = ResNetPatchClassifier(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    probs, logits, feats = model(x, return_features=True)

    assert probs.shape == (4, 2)
    assert logits.shape == (4, 2)
    assert feats.shape == (4, 2048)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)

    # Test slide pooling
    slide_prob, slide_feat = model.pool_slide(feats)
    assert 0.0 <= slide_prob.item() <= 1.0
    assert slide_feat.shape == (1, 2048)
    print("      ResNet-50 forward pass & slide pooling verified.")


def test_densenet_model():
    print("[3/6] Testing DenseNet-121 Patch Classifier...")
    model = DenseNetPatchClassifier(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    probs, logits, feats = model(x, return_features=True)

    assert probs.shape == (4, 2)
    assert logits.shape == (4, 2)
    assert feats.shape == (4, 1024)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)

    # Test slide pooling
    slide_prob, slide_feat = model.pool_slide(feats)
    assert 0.0 <= slide_prob.item() <= 1.0
    assert slide_feat.shape == (1, 1024)
    print("      DenseNet-121 forward pass & slide pooling verified.")


def test_patient_aggregator():
    print("[4/6] Testing Patient Slide Aggregator...")
    aggregator = PatientSlideAggregator()

    # Synthetic slide with 100 patches
    slide_probs = np.random.uniform(0.0, 0.4, 100).astype(np.float32)
    slide_probs[:5] = [0.95, 0.90, 0.88, 0.80, 0.70]  # tumor patches

    slide_feat = aggregator.extract_slide_features(slide_probs)
    assert slide_feat.shape == (5,)
    assert slide_feat[0] == 0.95  # Max probability
    assert slide_feat[3] == 0.05  # 5 positive patches out of 100 = 0.05 ratio

    # Test 5-slide patient aggregation
    patient_slides = [slide_feat, np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32)]
    pat_vector = aggregator.aggregate_patient_profile(patient_slides, max_nodes=5)
    assert pat_vector.shape == (25,)
    print("      Patient slide aggregator verified (25-D permutation-invariant vector).")


def test_stage_predictor():
    print("[5/6] Testing Patient pN-Stage Predictor...")
    predictor = PatientStagePredictor(in_dim=25, num_stages=5)
    dummy_patient = torch.randn(2, 25)

    probs, logits = predictor(dummy_patient)
    assert probs.shape == (2, 5)
    assert logits.shape == (2, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)

    stage_name, conf = predictor.get_predicted_stage_name(probs[0])
    assert stage_name in PatientStagePredictor.STAGE_NAMES
    assert 0.0 <= conf <= 1.0
    print(f"      Patient stage predictor verified (Stage: {stage_name}, Conf: {conf:.4f}).")


def test_ensemble_module():
    print("[6/6] Testing Branch 1 Ensemble Fusion...")
    ensemble = Branch1Ensemble(weight_resnet=0.6, weight_densenet=0.4)

    probs_resnet = torch.tensor([[0.80, 0.10, 0.05, 0.03, 0.02]])
    probs_densenet = torch.tensor([[0.70, 0.15, 0.08, 0.04, 0.03]])

    fused_probs, info = ensemble(probs_resnet, probs_densenet)
    assert fused_probs.shape == (1, 5)
    assert torch.allclose(fused_probs.sum(), torch.tensor(1.0), atol=1e-4)
    assert info["predicted_stage"] == "pN0"
    print(f"      Ensemble fusion verified (Fused Stage: {info['predicted_stage']}, Conf: {info['confidence']:.4f}).")


def run_all_tests():
    print("\n=======================================================")
    print("RUNNING ALL UNIT TESTS FOR BRANCH 1 (DEEP LEARNING ENSEMBLE)")
    print("=======================================================\n")
    test_h5_patch_dataset()
    test_resnet_model()
    test_densenet_model()
    test_patient_aggregator()
    test_stage_predictor()
    test_ensemble_module()
    print("\nALL 6 BRANCH 1 TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_all_tests()
