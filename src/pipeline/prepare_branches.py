"""
Multi-Branch Data Preparation Module.
Packages preprocessed WSI data into branch-specific formats:
- Branch 1: Deep Learning Ensemble (ResNet/DenseNet) -> HDF5 with patch images & binary labels.
- Branch 2: Histopathology & Context (CTransPath + Neighborhood Attention) -> PyTorch tensors with embeddings & (x, y) coordinates.
- Branch 3: Multimodal Analysis (PathDL + ClinicalML) -> Slide & patient-level metadata and burden tables.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import h5py
import torch
import pandas as pd


class BranchDataPackager:
    """
    Handles formatting and saving preprocessed WSI data for the 3 methodology branches.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.branch1_dir = self.output_dir / "branch1_patches"
        self.branch2_dir = self.output_dir / "branch2_features"
        self.branch3_dir = self.output_dir / "branch3_multimodal"

        for d in [self.branch1_dir, self.branch2_dir, self.branch3_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def export_branch1(
        self,
        slide_id: str,
        patches: List[np.ndarray],
        records: List[Dict[str, Any]],
    ) -> Path:
        """
        Exports patches and labels to HDF5 format for Branch 1 training.
        """
        out_path = self.branch1_dir / f"{slide_id}.h5"
        if len(patches) == 0:
            return out_path

        patch_array = np.stack(patches, axis=0)  # (N, H, W, 3) uint8
        coords = np.array([[r["x"], r["y"]] for r in records], dtype=np.int32)
        labels = np.array([r["label"] for r in records], dtype=np.int64)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("images", data=patch_array, compression="gzip", compression_opts=4)
            f.create_dataset("coords", data=coords)
            f.create_dataset("labels", data=labels)

        return out_path

    def export_branch2(
        self,
        slide_id: str,
        features: np.ndarray,
        records: List[Dict[str, Any]],
    ) -> Path:
        """
        Exports feature vectors and (x, y) spatial coordinates for Branch 2 (Selective Neighborhood Attention).
        """
        out_path = self.branch2_dir / f"{slide_id}.pt"
        coords = np.array([[r["x"], r["y"]] for r in records], dtype=np.int32)
        labels = np.array([r["label"] for r in records], dtype=np.int64)

        payload = {
            "slide_id": slide_id,
            "features": torch.from_numpy(features).float(),  # [N, D]
            "coords": torch.from_numpy(coords).long(),        # [N, 2]
            "labels": torch.from_numpy(labels).long(),        # [N]
        }
        torch.save(payload, out_path)
        return out_path

    def export_branch3_slide_summary(
        self,
        slide_id: str,
        patient_id: str,
        records: List[Dict[str, Any]],
        stage_labels_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Computes slide-level quantitative metastatic burden features for Branch 3.
        """
        total_patches = len(records)
        tumor_patches = sum(1 for r in records if r["label"] == 1)
        tumor_ratio = (tumor_patches / total_patches) if total_patches > 0 else 0.0

        slide_label = "unknown"
        patient_stage = "unknown"

        if stage_labels_df is not None:
            # Look up slide label
            slide_match = stage_labels_df[stage_labels_df["patient"] == f"{slide_id}.tif"]
            if not slide_match.empty:
                slide_label = slide_match.iloc[0]["stage"]

            # Look up patient pN stage
            pat_match = stage_labels_df[stage_labels_df["patient"] == f"{patient_id}.zip"]
            if not pat_match.empty:
                patient_stage = pat_match.iloc[0]["stage"]

        summary = {
            "patient_id": patient_id,
            "slide_id": slide_id,
            "total_tissue_patches": total_patches,
            "tumor_patches_annotated": tumor_patches,
            "annotated_tumor_burden_ratio": tumor_ratio,
            "ground_truth_slide_label": slide_label,
            "ground_truth_patient_stage": patient_stage,
        }

        # Save / append to CSV
        summary_csv = self.branch3_dir / "slides_multimodal_summary.csv"
        df_new = pd.DataFrame([summary])
        if summary_csv.exists():
            df_existing = pd.read_csv(summary_csv)
            # Remove duplicate slide entry if re-run
            df_existing = df_existing[df_existing["slide_id"] != slide_id]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(summary_csv, index=False)
        else:
            df_new.to_csv(summary_csv, index=False)

        return summary
