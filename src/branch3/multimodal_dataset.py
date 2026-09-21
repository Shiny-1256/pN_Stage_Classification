"""
Multimodal Dataset Builder for Branch 3.
Fuses Pathological Quantitative WSI Biomarkers with Patient Clinical Profiles
into a unified analytical matrix with ground-truth pN stage labels.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from .pathology_features import PathologyFeatureExtractor
from .clinical_features import ClinicalFeatureManager


class MultimodalDatasetBuilder:
    """
    Constructs the multimodal joint feature matrix.
    """

    STAGE_TO_INT = {
        "pN0": 0,
        "pN0(i+)": 1,
        "pN1mi": 2,
        "pN1": 3,
        "pN2": 4,
    }

    INT_TO_STAGE = {v: k for k, v in STAGE_TO_INT.items()}
    STAGE_NAMES = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]

    def __init__(
        self,
        summary_csv: Optional[str | Path] = None,
        meta_json: Optional[str | Path] = None,
        stage_csv: Optional[str | Path] = None,
    ):
        self.summary_csv = Path(summary_csv) if summary_csv else None
        self.meta_json = Path(meta_json) if meta_json else None
        self.stage_csv = Path(stage_csv) if stage_csv else None

        self.path_extractor = PathologyFeatureExtractor(self.summary_csv, self.meta_json)
        self.clin_manager = ClinicalFeatureManager()

    def build_dataset(
        self,
        patients: List[str],
        export_csv_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Merges pathological and clinical features, joins ground-truth stages, and returns a single DataFrame.
        """
        df_path = self.path_extractor.extract_cohort_features(
            patients=patients,
            summary_csv_path=self.summary_csv,
            meta_json_path=self.meta_json,
        )

        df_clin = self.clin_manager.get_cohort_dataframe(patients=patients)

        # Merge on patient_id
        df_merged = pd.merge(df_path, df_clin, on="patient_id", how="inner")

        # Load ground truth if stage_csv is available
        gt_stages = {}
        if self.stage_csv and self.stage_csv.exists():
            df_gt = pd.read_csv(self.stage_csv)
            for _, row in df_gt.iterrows():
                item = str(row["patient"]).strip()
                stage = str(row["stage"]).strip()
                if item.endswith(".zip"):
                    gt_stages[item.replace(".zip", "")] = stage

        df_merged["ground_truth_stage"] = df_merged["patient_id"].map(lambda p: gt_stages.get(p, "unknown"))
        df_merged["target_int"] = df_merged["ground_truth_stage"].map(lambda s: self.STAGE_TO_INT.get(s, -1))

        if export_csv_path:
            out_path = Path(export_csv_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_merged.to_csv(out_path, index=False)
            print(f"Exported multimodal patient matrix ({len(df_merged)} patients) to: {out_path}")

        return df_merged

    def get_features_and_targets(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Splits dataset into feature matrix X, target array y, and feature column names.
        """
        non_feature_cols = ["patient_id", "ground_truth_stage", "target_int"]
        feat_cols = [c for c in df.columns if c not in non_feature_cols]

        X = df[feat_cols].values.astype(np.float32)
        y = df["target_int"].values.astype(np.int64)

        return X, y, feat_cols
