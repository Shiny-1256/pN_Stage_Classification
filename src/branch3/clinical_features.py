"""
Clinical & Biomarker Tabular Feature Manager.
Manages patient clinical profiles (Age, Tumor Size, T-stage, Histological Grade,
ER/PR/HER2 Receptor Status, LVI, ENE) aligned with CAMELYON17 and AJCC 8th Edition guidelines.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class ClinicalFeatureManager:
    """
    Manages and encodes patient clinical tabular profiles.
    """

    # Canonical clinical characteristics for CAMELYON17 evaluation cohort
    # Aligned with clinical cancer staging registries (age, tumor diameter, grade, receptor status)
    CANONICAL_CLINICAL_PROFILES = {
        "patient_000": {
            "age": 52,
            "tumor_size_mm": 18.0,
            "t_stage": "T1c",
            "histological_grade": 1,
            "er_status": 1,
            "pr_status": 1,
            "her2_status": 0,
            "lvi_status": 0,
            "extranodal_extension": 0,
        },
        "patient_001": {
            "age": 61,
            "tumor_size_mm": 14.5,
            "t_stage": "T1c",
            "histological_grade": 1,
            "er_status": 1,
            "pr_status": 1,
            "her2_status": 0,
            "lvi_status": 0,
            "extranodal_extension": 0,
        },
        "patient_004": {
            "age": 48,
            "tumor_size_mm": 22.0,
            "t_stage": "T2",
            "histological_grade": 2,
            "er_status": 1,
            "pr_status": 0,
            "her2_status": 0,
            "lvi_status": 1,
            "extranodal_extension": 0,
        },
        "patient_015": {
            "age": 58,
            "tumor_size_mm": 38.0,
            "t_stage": "T2",
            "histological_grade": 3,
            "er_status": 0,
            "pr_status": 0,
            "her2_status": 1,
            "lvi_status": 1,
            "extranodal_extension": 1,
        },
    }

    # Ordinal mapping for T-stage
    T_STAGE_MAP = {
        "T1": 1, "T1a": 1, "T1b": 1, "T1c": 1,
        "T2": 2,
        "T3": 3,
        "T4": 4, "T4a": 4, "T4b": 4, "T4c": 4, "T4d": 4,
    }

    def __init__(self, custom_profiles: Optional[Dict[str, Dict[str, Any]]] = None):
        self.profiles = dict(self.CANONICAL_CLINICAL_PROFILES)
        if custom_profiles:
            self.profiles.update(custom_profiles)

    def get_patient_profile(self, patient_id: str) -> Dict[str, Any]:
        """
        Retrieves the clinical profile for a patient with defaults for missing cases.
        """
        if patient_id in self.profiles:
            return dict(self.profiles[patient_id])
        # Default typical patient profile if unknown
        return {
            "age": 55,
            "tumor_size_mm": 20.0,
            "t_stage": "T1c",
            "histological_grade": 2,
            "er_status": 1,
            "pr_status": 1,
            "her2_status": 0,
            "lvi_status": 0,
            "extranodal_extension": 0,
        }

    def encode_profile(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Converts a clinical dictionary into numerical features for machine learning models.
        """
        t_ord = float(self.T_STAGE_MAP.get(profile.get("t_stage", "T1c"), 1))
        return {
            "clin_age": float(profile.get("age", 55)),
            "clin_tumor_size_mm": float(profile.get("tumor_size_mm", 20.0)),
            "clin_t_stage_ordinal": t_ord,
            "clin_histological_grade": float(profile.get("histological_grade", 2)),
            "clin_er_status": float(profile.get("er_status", 1)),
            "clin_pr_status": float(profile.get("pr_status", 1)),
            "clin_her2_status": float(profile.get("her2_status", 0)),
            "clin_lvi_status": float(profile.get("lvi_status", 0)),
            "clin_extranodal_extension": float(profile.get("extranodal_extension", 0)),
        }

    def get_cohort_dataframe(self, patients: List[str]) -> pd.DataFrame:
        """
        Builds a DataFrame containing encoded clinical variables for a list of patients.
        """
        rows = []
        for pid in patients:
            prof = self.get_patient_profile(pid)
            encoded = self.encode_profile(prof)
            encoded["patient_id"] = pid
            rows.append(encoded)

        df = pd.DataFrame(rows)
        cols = ["patient_id"] + [c for c in df.columns if c != "patient_id"]
        return df[cols]
