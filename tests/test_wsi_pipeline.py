"""
Unit and Smoke Test for WSI Preprocessing Pipeline.
Tests slide reading, tissue mask generation, coordinate mapping, and package export.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np

from src.utils.wsi_reader import WSIReader
from src.utils.annotation_parser import AnnotationParser
from src.preprocessing.patch_extractor import PatchExtractor
from src.preprocessing.stain_normalizer import MacenkoNormalizer
from src.pipeline.prepare_branches import BranchDataPackager


def test_wsi_pipeline_smoke():
    slide_path = Path(r"C:\Users\Shiny\.gemini\antigravity\scratch\dataset\raw_wsi\patient_000\patient_000_node_0.tif")
    assert slide_path.exists(), f"Sample slide not found: {slide_path}"

    print("[1/5] Testing WSI Reader...")
    with WSIReader(slide_path) as reader:
        assert reader.width > 0 and reader.height > 0
        print(f"      Dimensions: {reader.width} x {reader.height}, Levels: {reader.num_levels}")

        print("[2/5] Testing Thumbnail & Tissue Segmentation...")
        extractor = PatchExtractor(patch_size=224, min_tissue_ratio=0.50, thumbnail_level=7)
        mask, records = extractor.generate_patch_coordinates(reader)
        assert mask.ndim == 2
        print(f"      Mask shape: {mask.shape}, Valid tissue patches: {len(records)}")

        assert len(records) > 0, "Expected positive count of tissue patches"

        print("[3/5] Testing Level 0 Patch Extraction...")
        first_record = records[0]
        patch = reader.read_region(
            location=(first_record["x"], first_record["y"]),
            level=0,
            size=(224, 224),
        )
        assert patch.shape == (224, 224, 3)
        assert patch.dtype == np.uint8
        print(f"      Extracted patch at ({first_record['x']}, {first_record['y']}): shape={patch.shape}")

        print("[4/5] Testing Macenko Stain Normalization...")
        normalizer = MacenkoNormalizer()
        norm_patch = normalizer.normalize(patch)
        assert norm_patch.shape == (224, 224, 3)
        print("      Stain normalization successful.")

        print("[5/5] Testing Multi-Branch Exporter...")
        test_out_dir = Path(r"C:\Users\Shiny\.gemini\antigravity\scratch\pN_Stage_Classification\output\test_run")
        packager = BranchDataPackager(test_out_dir)

        # Export test subset of 5 patches
        sample_patches = [patch]
        sample_records = [first_record]
        fake_features = np.random.randn(1, 2048).astype(np.float32)

        h5_p = packager.export_branch1("test_slide", sample_patches, sample_records)
        pt_p = packager.export_branch2("test_slide", fake_features, sample_records)
        summary = packager.export_branch3_slide_summary("test_slide", "patient_000", sample_records)

        assert h5_p.exists()
        assert pt_p.exists()
        print(f"      Branch 1 (.h5): {h5_p.name} ({h5_p.stat().st_size} bytes)")
        print(f"      Branch 2 (.pt): {pt_p.name} ({pt_p.stat().st_size} bytes)")
        print(f"      Branch 3 summary recorded: {summary}")

    print("\nALL SMOKE TESTS PASSED!")


if __name__ == "__main__":
    test_wsi_pipeline_smoke()
