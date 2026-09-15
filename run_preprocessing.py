"""
Main Preprocessing Runner for pN-Stage Classification Pipeline.
Processes WSIs, extracts patches, performs stain normalization, extracts deep embeddings,
and packages data for Branch 1, Branch 2, and Branch 3.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import cv2
import pandas as pd
from tqdm import tqdm

from src.utils.wsi_reader import WSIReader
from src.utils.annotation_parser import AnnotationParser
from src.preprocessing.patch_extractor import PatchExtractor
from src.preprocessing.stain_normalizer import MacenkoNormalizer
from src.feature_extraction.patch_encoder import PatchEncoder
from src.pipeline.prepare_branches import BranchDataPackager


def process_slide(
    slide_path: Path,
    config: dict,
    packager: BranchDataPackager,
    encoder: PatchEncoder,
    normalizer: MacenkoNormalizer,
    extractor: PatchExtractor,
    stage_labels_df: pd.DataFrame,
    save_masks: bool = True,
    max_patches: int = None,
):
    slide_id = slide_path.stem
    # Identify patient name (e.g., patient_000 from patient_000_node_0)
    parts = slide_id.split("_")
    patient_id = f"{parts[0]}_{parts[1]}"

    print(f"\n=======================================================", flush=True)
    print(f"Processing Slide: {slide_id} (Patient: {patient_id})", flush=True)
    print(f"=======================================================", flush=True)

    # Look for corresponding annotation XML
    ann_dir = Path(config["dataset"]["annotations_dir"])
    xml_path = ann_dir / f"{slide_id}.xml"
    annotation_parser = None
    if xml_path.exists():
        print(f" Found lesion annotation: {xml_path.name}", flush=True)
        annotation_parser = AnnotationParser(xml_path)
    else:
        print(f"  No XML lesion annotation for {slide_id} (Slide marked negative / unannotated)", flush=True)

    # Open WSI
    with WSIReader(slide_path) as reader:
        print(f"  WSI Dimensions: {reader.width} x {reader.height} | Levels: {reader.num_levels}", flush=True)

        # Generate tissue mask & patch coordinates
        print("  Generating tissue mask and coordinate grid...", flush=True)
        mask, records = extractor.generate_patch_coordinates(
            wsi_reader=reader,
            annotation_parser=annotation_parser,
        )
        print(f"  Identified {len(records)} valid tissue patches.", flush=True)

        # Save tissue mask visualization
        if save_masks:
            mask_out_dir = packager.output_dir / "tissue_masks"
            mask_out_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_out_dir / f"{slide_id}_mask.png"
            cv2.imwrite(str(mask_path), mask * 255)

        if len(records) == 0:
            print("  Warning: No tissue detected on this slide!", flush=True)
            return

        # Optional cap on patches for quick testing
        if max_patches and len(records) > max_patches:
            print(f"  Limiting to {max_patches} patches for testing...", flush=True)
            records = records[:max_patches]

        # Extract patch images from WSI
        print("  Extracting Level 0 patch tiles...", flush=True)
        patches = []
        for r in tqdm(records, desc="Extracting & Normalizing"):
            patch = reader.read_region(
                location=(r["x"], r["y"]),
                level=0,
                size=(extractor.patch_size, extractor.patch_size),
            )
            # Stain Normalization
            if config["stain_normalization"]["enabled"]:
                patch = normalizer.normalize(patch)
            patches.append(patch)

        # Feature Extraction for Branch 2 & 3
        print("  Extracting deep feature representations...", flush=True)
        features = encoder.extract_features(
            patches=patches,
            batch_size=config["feature_extraction"]["batch_size"],
        )
        print(f"  Feature shape: {features.shape}", flush=True)

        # Packaging
        print("  Exporting Branch 1 (HDF5 patches)...", flush=True)
        h5_path = packager.export_branch1(slide_id, patches, records)

        print("  Exporting Branch 2 (PyTorch tensors + coordinates)...", flush=True)
        pt_path = packager.export_branch2(slide_id, features, records)

        print("  Exporting Branch 3 (Slide/Patient summary metrics)...", flush=True)
        summary = packager.export_branch3_slide_summary(
            slide_id=slide_id,
            patient_id=patient_id,
            records=records,
            stage_labels_df=stage_labels_df,
        )

        print(f"Completed {slide_id}: {summary['total_tissue_patches']} patches | "
              f"Annotated Tumors: {summary['tumor_patches_annotated']} | "
              f"Stage: {summary['ground_truth_patient_stage']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="CAMELYON17 WSI Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--patient", type=str, default=None, help="Process specific patient (e.g. patient_000)")
    parser.add_argument("--slide", type=str, default=None, help="Process specific slide (e.g. patient_000_node_0.tif)")
    parser.add_argument("--max_patches", type=int, default=None, help="Cap patches per slide (for smoke testing)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load stage labels master table
    labels_csv = Path(config["dataset"]["stage_labels_csv"])
    stage_labels_df = pd.read_csv(labels_csv) if labels_csv.exists() else None

    # Initialize modules
    packager = BranchDataPackager(config["dataset"]["output_dir"])
    encoder = PatchEncoder(
        model_name=config["feature_extraction"]["encoder"],
        use_amp=config["feature_extraction"]["use_amp"],
    )
    normalizer = MacenkoNormalizer()
    extractor = PatchExtractor(
        patch_size=config["preprocessing"]["patch_size"],
        min_tissue_ratio=config["preprocessing"]["min_tissue_ratio"],
        thumbnail_level=config["preprocessing"].get("mask_downsample_level", 7),
    )

    # Gather slides to process
    raw_wsi_dir = Path(config["dataset"]["raw_wsi_dir"])
    slides_to_process = []

    if args.slide:
        match = list(raw_wsi_dir.glob(f"**/{args.slide}"))
        if match:
            slides_to_process.append(match[0])
    elif args.patient:
        pat_dir = raw_wsi_dir / args.patient
        if pat_dir.exists():
            slides_to_process.extend(sorted(list(pat_dir.glob("*.tif"))))
    else:
        slides_to_process = sorted(list(raw_wsi_dir.glob("*/*.tif")))

    print(f"Found {len(slides_to_process)} slide(s) to process.")

    for slide_p in slides_to_process:
        process_slide(
            slide_path=slide_p,
            config=config,
            packager=packager,
            encoder=encoder,
            normalizer=normalizer,
            extractor=extractor,
            stage_labels_df=stage_labels_df,
            max_patches=args.max_patches,
        )


if __name__ == "__main__":
    main()
