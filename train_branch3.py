"""
Training & Evaluation Pipeline for Branch 3 (Multimodal PathDL + ClinicalML).
Constructs joint patient feature matrices, trains the gradient-boosted multimodal model,
and outputs patient-level predictions and feature importances.
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.branch3.multimodal_dataset import MultimodalDatasetBuilder
from src.branch3.multimodal_model import MultimodalStagingClassifier


def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate Branch 3 Multimodal Model")
    parser.add_argument("--model_type", type=str, default="random_forest", choices=["random_forest", "xgboost", "gradient_boosting"])
    args = parser.parse_args()

    print("=" * 70)
    print("BRANCH 3: MULTIMODAL PATHDL + CLINICALML PIPELINE")
    print("=" * 70)

    summary_csv = PROJECT_ROOT / "output" / "branch3_multimodal" / "slides_multimodal_summary.csv"
    meta_json = PROJECT_ROOT / "output" / "slide_visualizations" / "slide_meta.json"
    stage_csv = PROJECT_ROOT.parent / "dataset" / "stage_labels.csv"
    out_dir = PROJECT_ROOT / "output" / "branch3_multimodal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4 Genuine CAMELYON17 Patients
    patients = ["patient_000", "patient_001", "patient_004", "patient_015"]
    print(f"Building multimodal matrix for {len(patients)} genuine cohort patients: {patients}")

    builder = MultimodalDatasetBuilder(
        summary_csv=summary_csv,
        meta_json=meta_json,
        stage_csv=stage_csv,
    )

    matrix_path = out_dir / "multimodal_patient_matrix.csv"
    df_matrix = builder.build_dataset(patients=patients, export_csv_path=matrix_path)

    X, y, feat_cols = builder.get_features_and_targets(df_matrix)
    print(f"Dataset shape: {X.shape[0]} patients, {X.shape[1]} multimodal features")
    print(f"Feature names ({len(feat_cols)}): {feat_cols}")

    # Train Multimodal Model
    print(f"\nTraining {args.model_type.upper()} classifier...")
    classifier = MultimodalStagingClassifier(model_type=args.model_type, random_state=42)
    classifier.fit(X, y, feature_names=feat_cols)

    # Generate calibrated 5-class posteriors
    probs = classifier.predict_proba(X)
    pred_stages = classifier.predict(X)
    feat_importance = classifier.get_feature_importance()

    predictions_dict = {}
    print("\n" + "=" * 70)
    print("BRANCH 3 PATIENT PREDICTIONS & EVALUATION:")
    print("=" * 70)

    for i, pid in enumerate(patients):
        gt_stage = df_matrix.iloc[i]["ground_truth_stage"]
        pred_stage = pred_stages[i]
        conf = float(probs[i, classifier.STAGE_TO_INT[pred_stage]])
        p_dist = {s: float(probs[i, idx]) for idx, s in enumerate(classifier.STAGE_NAMES)}

        match_str = "[MATCH]" if pred_stage.lower() == gt_stage.lower() else "[DISCREPANCY]"
        print(f"Patient {pid}:")
        print(f"  • Predicted Stage: {pred_stage} (Confidence: {conf*100:.1f}%)")
        print(f"  • Ground Truth:    {gt_stage}  {match_str}")
        print(f"  • Posteriors:      { {k: round(v, 3) for k, v in p_dist.items()} }")

        predictions_dict[pid] = {
            "patient_id": pid,
            "predicted_stage": pred_stage,
            "ground_truth_stage": gt_stage,
            "confidence": conf,
            "stage_probabilities": p_dist,
            "is_concordant": (pred_stage.lower() == gt_stage.lower()),
        }

    # Top Features
    print("\nTop Multimodal Feature Importances:")
    for k, v in list(feat_importance.items())[:6]:
        print(f"  - {k:<28}: {v*100:.2f}%")

    # Save outputs
    pred_json_path = out_dir / "branch3_patient_predictions.json"
    with open(pred_json_path, "w") as fp:
        json.dump(predictions_dict, fp, indent=2)
    print(f"\nSaved Branch 3 predictions to: {pred_json_path}")

    model_path = out_dir / "branch3_model.joblib"
    classifier.save(model_path)

    # Also save feature importance JSON
    imp_json_path = out_dir / "branch3_feature_importance.json"
    with open(imp_json_path, "w") as fp:
        json.dump(feat_importance, fp, indent=2)


if __name__ == "__main__":
    main()
