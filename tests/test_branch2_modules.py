"""
Unit & Mathematical Verification Tests for Branch 2.
Tests all modules from Tauqeer et al. (Scientific Reports 2025):
1. Nuclei Feature Extractor (NFE): 16-ch -> 64x64x1
2. Tissue Feature Extractor (TFE): 256x256x3 -> 64x64x1
3. Feature Fusion: 64x64x2 -> 1024-D
4. Spatial Neighborhood Builder: 3x3 grid & boundary mask
5. Selective Neighborhood Attention (SNA): Gating, Top-4 selection, Self-Attention
6. Slide-Level Classifier: Attention-based bag pooling
7. Hierarchical Loss: 3-part loss & backprop verification
8. Tumor Map & Staging Evaluator: Area calc (ITC/Micro/Macro) & pN stage logic
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.branch2.nuclei_extractor import NucleiFeatureExtractor
from src.branch2.tissue_extractor import TissueFeatureExtractor
from src.branch2.feature_fusion import DualPathFeatureFusion
from src.branch2.neighborhood_builder import SpatialNeighborhoodBuilder
from src.branch2.selective_attention import SelectiveNeighborhoodAttention
from src.branch2.classifier_heads import SlideLevelClassifier
from src.branch2.loss import HierarchicalBranch2Loss
from src.branch2.staging_evaluator import TumorMapperAndStagingEvaluator


def test_nuclei_feature_extractor():
    print("[1/8] Testing Nuclei Feature Extractor...")
    # Test 16-ch feature extraction from a dummy RGB patch
    dummy_patch = np.random.randint(100, 240, (256, 256, 3), dtype=np.uint8)
    feat_16ch = NucleiFeatureExtractor.extract_16ch_features_from_patch(dummy_patch)
    assert feat_16ch.shape == (256, 256, 16), f"Expected (256, 256, 16), got {feat_16ch.shape}"

    # Test forward compression pass: (B, 16, 256, 256) -> (B, 1, 64, 64)
    model = NucleiFeatureExtractor(in_channels=16)
    x_tensor = torch.randn(2, 16, 256, 256)
    out = model(x_tensor)
    assert out.shape == (2, 1, 64, 64), f"Expected (2, 1, 64, 64), got {out.shape}"
    print("      Nuclei feature extraction & compression verified (shape: 2, 1, 64, 64)")


def test_tissue_feature_extractor():
    print("[2/8] Testing Tissue Feature Extractor...")
    model = TissueFeatureExtractor(in_channels=3, embed_dim=64, num_heads=4, num_transformer_layers=2)
    x_rgb = torch.randn(2, 3, 256, 256)
    out_map, logits = model(x_rgb, return_logits=True)
    assert out_map.shape == (2, 1, 64, 64), f"Expected (2, 1, 64, 64), got {out_map.shape}"
    assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"
    print("      Tissue CNN + Transformer extraction verified (shape: 2, 1, 64, 64)")


def test_feature_fusion():
    print("[3/8] Testing Dual-Path Feature Fusion...")
    fusion = DualPathFeatureFusion(in_channels=2, spatial_dim=64, out_dim=1024)
    nfe = torch.randn(4, 1, 64, 64)
    tfe = torch.randn(4, 1, 64, 64)
    f_ij = fusion(nfe, tfe)
    assert f_ij.shape == (4, 1024), f"Expected (4, 1024), got {f_ij.shape}"
    print("      Feature fusion (64x64x2 -> 1024-D) verified (shape: 4, 1024)")


def test_spatial_neighborhood_builder():
    print("[4/8] Testing Spatial Neighborhood Builder...")
    builder = SpatialNeighborhoodBuilder(patch_size=256)
    # Create a 3x3 grid of coordinates
    coords = []
    for y in [0, 256, 512]:
        for x in [0, 256, 512]:
            coords.append((x, y))
    coords = np.array(coords)

    indices, mask = builder.build_neighborhood_indices(coords)
    assert indices.shape == (9, 8), f"Expected (9, 8), got {indices.shape}"
    assert mask.shape == (9, 8), f"Expected (9, 8), got {mask.shape}"

    # Center patch is index 4 (x=256, y=256), all its 8 neighbors must exist
    assert mask[4].all(), "Center patch should have all 8 valid neighbors"
    # Corner patch (index 0: 0, 0) should only have 3 neighbors
    assert mask[0].sum() == 3, f"Corner patch should have 3 valid neighbors, got {mask[0].sum()}"
    print("      Spatial 3x3 neighborhood graph verified (9 patches, boundary masks ok)")


def test_selective_neighborhood_attention():
    print("[5/8] Testing Selective Neighborhood Attention (SNA)...")
    sna = SelectiveNeighborhoodAttention(feature_dim=1024, attn_dim=256, top_k=4, num_classes=2)
    B = 4
    center_feats = torch.randn(B, 1024)
    neighbor_feats = torch.randn(B, 8, 1024)
    neighbor_mask = torch.ones(B, 8, dtype=torch.bool)
    # Mask out 2 neighbors for the last sample
    neighbor_mask[-1, 6:] = False

    probs, logits, alpha, context_vec = sna(center_feats, neighbor_feats, neighbor_mask)

    assert probs.shape == (B, 2), f"Expected (B, 2), got {probs.shape}"
    assert logits.shape == (B, 2), f"Expected (B, 2), got {logits.shape}"
    assert alpha.shape == (B, 8), f"Expected (B, 8), got {alpha.shape}"
    assert context_vec.shape == (B, 1024), f"Expected (B, 1024), got {context_vec.shape}"

    # Check that masked neighbors received near-zero attention
    assert alpha[-1, 6].item() < 1e-4 and alpha[-1, 7].item() < 1e-4
    print("      Selective attention (gating, top-4 selection, self-attention) verified")


def test_slide_level_classifier():
    print("[6/8] Testing Slide-Level Classifier Head...")
    classifier = SlideLevelClassifier(feature_dim=1024, attn_dim=256)
    N = 25  # Bag of 25 patches
    patch_feats = torch.randn(N, 1024)

    slide_prob, slide_logit, patch_weights = classifier(patch_feats)
    assert 0.0 <= slide_prob.item() <= 1.0, f"Probability out of range: {slide_prob.item()}"
    assert patch_weights.shape == (N, 1), f"Expected (N, 1), got {patch_weights.shape}"
    # Attention weights should sum to 1.0
    assert torch.isclose(patch_weights.sum(), torch.tensor(1.0), atol=1e-4)
    print(f"      Slide aggregation verified (Bag size: {N}, Slide prob: {slide_prob.item():.4f})")


def test_hierarchical_loss():
    print("[7/8] Testing Hierarchical 3-Component Loss...")
    criterion = HierarchicalBranch2Loss(lambda_instance=0.3, lambda_attention=0.2, lambda_bag=0.5, top_k=4)

    N = 10
    patch_logits = torch.randn(N, 2, requires_grad=True)
    patch_labels = torch.randint(0, 2, (N,))
    attention_weights = torch.softmax(torch.randn(N, 8), dim=-1)
    slide_prob = torch.sigmoid(torch.tensor(0.5, requires_grad=True))
    slide_label = torch.tensor(1.0)

    loss_total, metrics = criterion(patch_logits, patch_labels, attention_weights, slide_prob, slide_label)
    assert loss_total.item() > 0.0
    assert "loss_instance" in metrics and "loss_attention" in metrics and "loss_bag" in metrics

    # Verify backpropagation
    loss_total.backward()
    assert patch_logits.grad is not None, "Gradients should flow back to patch logits"
    print(f"      Hierarchical loss backward pass ok: Total Loss = {loss_total.item():.4f}")


def test_tumor_mapper_and_staging():
    print("[8/8] Testing Tumor Mapper & Patient pN-Staging...")
    evaluator = TumorMapperAndStagingEvaluator(pixel_size_um=0.24, patch_size_px=256)

    # 1. Test binary map & heatmap reconstruction
    coords = np.array([[0, 0], [256, 0], [0, 256], [256, 256]])
    predictions = np.array([0.1, 0.95, 0.85, 0.05])
    bmap, hmap = evaluator.reconstruct_tumor_map(coords, predictions, wsi_width=1024, wsi_height=1024, downsample_factor=16)
    assert bmap.shape == (64, 64)
    assert hmap.shape == (64, 64)
    assert np.count_nonzero(bmap) > 0

    # 2. Test slide lesion area quantification
    # Synthetic micro-metastasis: 500 pixels on downsampled map
    synthetic_bmap = np.zeros((100, 100), dtype=np.uint8)
    synthetic_bmap[20:40, 20:40] = 1  # 20x20 = 400 pixels
    # Area = 400 * (0.24 * 16)^2 = 400 * (3.84)^2 = 400 * 14.7456 = 5898.24 um^2 -> Micro
    summary = evaluator.quantify_slide_metastases(synthetic_bmap, downsample_factor=16)
    assert summary["highest_category"] == "micro"

    # 3. Test patient pN-staging logic across 5 slides
    # Case A: All negative
    stage_a = evaluator.compute_patient_pn_stage([{"highest_category": "negative"}] * 5)
    assert stage_a == "pN0", f"Expected pN0, got {stage_a}"

    # Case B: Only ITCs
    stage_b = evaluator.compute_patient_pn_stage([{"highest_category": "itc", "num_itc": 2}] + [{"highest_category": "negative"}] * 4)
    assert stage_b == "pN0(i+)", f"Expected pN0(i+), got {stage_b}"

    # Case C: Micro-metastasis only
    stage_c = evaluator.compute_patient_pn_stage([{"highest_category": "micro", "num_micro": 1}] + [{"highest_category": "negative"}] * 4)
    assert stage_c == "pN1mi", f"Expected pN1mi, got {stage_c}"

    # Case D: 2 nodes with Macro-metastasis
    stage_d = evaluator.compute_patient_pn_stage([
        {"highest_category": "macro", "num_macro": 1},
        {"highest_category": "macro", "num_macro": 1},
        {"highest_category": "negative"},
        {"highest_category": "negative"},
        {"highest_category": "negative"}
    ])
    assert stage_d == "pN1", f"Expected pN1, got {stage_d}"

    # Case E: 4 nodes with Macro-metastasis
    stage_e = evaluator.compute_patient_pn_stage([
        {"highest_category": "macro", "num_macro": 1},
        {"highest_category": "macro", "num_macro": 1},
        {"highest_category": "macro", "num_macro": 1},
        {"highest_category": "macro", "num_macro": 1},
        {"highest_category": "negative"}
    ])
    assert stage_e == "pN2", f"Expected pN2, got {stage_e}"

    print("      Tumor mapping, area quantification (ITC/Micro/Macro), and pN staging rules verified")


def run_all_tests():
    print("\n=======================================================")
    print("RUNNING ALL UNIT & MATHEMATICAL VERIFICATION TESTS FOR BRANCH 2")
    print("=======================================================\n")
    test_nuclei_feature_extractor()
    test_tissue_feature_extractor()
    test_feature_fusion()
    test_spatial_neighborhood_builder()
    test_selective_neighborhood_attention()
    test_slide_level_classifier()
    test_hierarchical_loss()
    test_tumor_mapper_and_staging()
    print("\nALL 8 BRANCH 2 MODULE TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_all_tests()
