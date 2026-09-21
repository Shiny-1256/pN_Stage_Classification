"""
Quantitative Pathological Biomarker & Burden Feature Extractor.
Extracts mathematical lesion burden descriptors from whole-slide summary records and metadata.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd


class PathologyFeatureExtractor:
    """
    Computes patient-level quantitative pathology features from slide-level summaries.
    """

    def __init__(self, summary_csv: Optional[str | Path] = None, meta_json: Optional[str | Path] = None):
        self.summary_csv = Path(summary_csv) if summary_csv else None
        self.meta_json = Path(meta_json) if meta_json else None

    def extract_features_for_patient(
        self,
        patient_id: str,
        slide_summaries: List[Dict[str, Any]],
        slide_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Computes 12 quantitative WSI burden metrics for a patient across all examined lymph nodes.
        """
        if not slide_summaries:
            return {
                "cumulative_tumor_burden": 0.0,
                "max_nodal_burden": 0.0,
                "mean_nodal_burden": 0.0,
                "std_nodal_burden": 0.0,
                "pos_nodes_ratio": 0.0,
                "pos_nodes_count": 0.0,
                "total_tissue_patches": 0.0,
                "total_tumor_patches": 0.0,
                "max_patch_probability": 0.0,
                "num_macro_lesions": 0.0,
                "num_micro_lesions": 0.0,
                "num_itc_lesions": 0.0,
                "nodal_burden_entropy": 0.0,
            }

        # Nodal burden ratios
        ratios = [float(s.get("annotated_tumor_burden_ratio", 0.0)) for s in slide_summaries]
        total_tissue = sum(int(s.get("total_tissue_patches", 0)) for s in slide_summaries)
        total_tumor = sum(int(s.get("tumor_patches_annotated", 0)) for s in slide_summaries)

        pos_nodes = sum(1 for r in ratios if r > 0.0)

        # Lesion counts & max probabilities from slide_metadata
        macro_c = 0
        micro_c = 0
        itc_c = 0
        max_prob = 0.0

        if slide_metadata:
            for s in slide_summaries:
                sid = s.get("slide_id", "")
                m = slide_metadata.get(sid, {})
                macro_c += int(m.get("num_macro", 0))
                micro_c += int(m.get("num_micro", 0))
                itc_c += int(m.get("num_itc", 0))
                max_prob = max(max_prob, float(m.get("max_prob", 0.0)))
        else:
            # Fallback based on ground truth if metadata not supplied
            for s in slide_summaries:
                lbl = str(s.get("ground_truth_slide_label", "")).lower()
                if "macro" in lbl:
                    macro_c += 1
                elif "micro" in lbl:
                    micro_c += 1
                elif "itc" in lbl:
                    itc_c += 1

        # Entropy of burden across nodes (measures spatial dissemination)
        eps = 1e-7
        p_arr = np.array(ratios) + eps
        p_norm = p_arr / p_arr.sum()
        entropy = -float(np.sum(p_norm * np.log(p_norm)))

        return {
            "cumulative_tumor_burden": float(sum(ratios)),
            "max_nodal_burden": float(max(ratios)) if ratios else 0.0,
            "mean_nodal_burden": float(np.mean(ratios)) if ratios else 0.0,
            "std_nodal_burden": float(np.std(ratios)) if ratios else 0.0,
            "pos_nodes_ratio": float(pos_nodes / max(1, len(slide_summaries))),
            "pos_nodes_count": float(pos_nodes),
            "total_tissue_patches": float(total_tissue),
            "total_tumor_patches": float(total_tumor),
            "max_patch_probability": float(max_prob),
            "num_macro_lesions": float(macro_c),
            "num_micro_lesions": float(micro_c),
            "num_itc_lesions": float(itc_c),
            "nodal_burden_entropy": entropy,
        }

    def extract_cohort_features(
        self,
        patients: List[str],
        summary_csv_path: Optional[str | Path] = None,
        meta_json_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Constructs a DataFrame of pathological features for all specified patients.
        """
        csv_path = Path(summary_csv_path) if summary_csv_path else self.summary_csv
        meta_path = Path(meta_json_path) if meta_json_path else self.meta_json

        df_summary = pd.DataFrame()
        if csv_path and csv_path.exists():
            df_summary = pd.read_csv(csv_path)

        meta_dict = {}
        if meta_path and meta_path.exists():
            with open(meta_path, "r") as fp:
                meta_dict = json.load(fp)

        records = []
        for pid in patients:
            p_slides = []
            if not df_summary.empty:
                p_slides = df_summary[df_summary["patient_id"] == pid].to_dict(orient="records")

            feats = self.extract_features_for_patient(pid, p_slides, meta_dict)
            feats["patient_id"] = pid
            records.append(feats)

        df_out = pd.DataFrame(records)
        # Put patient_id first
        cols = ["patient_id"] + [c for c in df_out.columns if c != "patient_id"]
        return df_out[cols]
