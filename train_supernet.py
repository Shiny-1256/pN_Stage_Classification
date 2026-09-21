"""
SuperNet Late Fusion Training & Consensus Prediction Pipeline.
Integrates Branch 1 (Cellular-Tissue Ensemble), Branch 2 (Selective Neighborhood Attention),
and Branch 3 (Multimodal PathDL + ClinicalML) into an optimal clinical consensus decision.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.late_fusion_model import SuperNetConsensusClassifier


def main():
    print("=" * 70)
    print("SUPERNET: MULTI-BRANCH LATE FUSION META-LEARNER")
    print("=" * 70)

    b1_path = PROJECT_ROOT / "output" / "branch1_predictions" / "branch1_patient_predictions.json"
    b2_path = PROJECT_ROOT / "output" / "branch2_predictions" / "branch2_patient_predictions.json"
    b3_path = PROJECT_ROOT / "output" / "branch3_multimodal" / "branch3_patient_predictions.json"
    stage_csv = PROJECT_ROOT.parent / "dataset" / "stage_labels.csv"

    out_dir = PROJECT_ROOT / "output" / "final_fusion_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all branch predictions
    with open(b1_path, "r") as fp:
        raw_b1 = json.load(fp)
        b1_dict = {item["patient_id"]: item for item in raw_b1}

    with open(b2_path, "r") as fp:
        b2_dict = json.load(fp)

    with open(b3_path, "r") as fp:
        b3_dict = json.load(fp)

    gt_dict = {}
    if stage_csv.exists():
        df_gt = pd.read_csv(stage_csv)
        for _, row in df_gt.iterrows():
            item = str(row["patient"]).strip()
            stage = str(row["stage"]).strip()
            if item.endswith(".zip"):
                gt_dict[item.replace(".zip", "")] = stage

    patients = sorted(list(set(b1_dict.keys()) & set(b2_dict.keys()) & set(b3_dict.keys())))
    print(f"Aggregating predictions for {len(patients)} patients: {patients}")

    supernet = SuperNetConsensusClassifier(
        weights={"branch1": 0.30, "branch2": 0.40, "branch3": 0.30}
    )

    consensus_results = {}
    print("\n" + "=" * 70)
    print("SUPERNET CONSENSUS STAGING EVALUATION:")
    print("=" * 70)

    for pid in patients:
        p1 = b1_dict.get(pid, {})
        p2 = b2_dict.get(pid, {})
        p3 = b3_dict.get(pid, {})
        gt_stage = gt_dict.get(pid, "unknown")

        dist1 = p1.get("stage_probabilities", {})
        # For Branch 2: convert predicted stage or slide lesion breakdown into distribution
        b2_stage = p2.get("predicted_pn_stage", "pN0")
        dist2 = {s: 0.05 for s in supernet.STAGE_NAMES}
        dist2[b2_stage] = 0.80
        total2 = sum(dist2.values())
        dist2 = {s: v / total2 for s, v in dist2.items()}

        dist3 = p3.get("stage_probabilities", {})

        c_stage, c_conf, c_dist, audit = supernet.fuse_probabilities(dist1, dist2, dist3)
        is_match = (c_stage.lower() == gt_stage.lower())
        match_str = "[MATCH]" if is_match else "[DISCREPANCY]"

        print(f"Patient {pid}:")
        print(f"  • Branch 1 (Cellular-Tissue):   {audit['branch1_pred']}")
        print(f"  • Branch 2 (Neighborhood Attn): {audit['branch2_pred']}")
        print(f"  • Branch 3 (Multimodal ML):     {audit['branch3_pred']}")
        print(f"  • SuperNet Consensus Stage:     {c_stage} (Confidence: {c_conf*100:.1f}%)")
        print(f"  • Ground Truth (CAMELYON17):   {gt_stage}  {match_str}")
        print(f"  • Multi-Branch Agreement:       {audit['agreement_ratio']*100:.0f}% ({'Unanimous' if audit['all_branches_unanimous'] else 'Majority Vote'})")
        print("-" * 70)

        consensus_results[pid] = {
            "patient_id": pid,
            "consensus_stage": c_stage,
            "confidence": c_conf,
            "ground_truth_stage": gt_stage,
            "is_concordant": is_match,
            "stage_probabilities": c_dist,
            "branch_breakdown": {
                "branch1": {"stage": audit["branch1_pred"], "confidence": p1.get("confidence", 0.0)},
                "branch2": {"stage": audit["branch2_pred"], "positive_nodes": p2.get("positive_nodes", 0)},
                "branch3": {"stage": audit["branch3_pred"], "confidence": p3.get("confidence", 0.0)},
            },
            "audit_trail": audit,
        }

    # Export consensus JSON
    out_json = out_dir / "supernet_consensus_predictions.json"
    with open(out_json, "w") as fp:
        json.dump(consensus_results, fp, indent=2)
    print(f"\nSaved SuperNet consensus predictions to: {out_json}")

    # Save model checkpoint
    model_file = out_dir / "supernet_model.joblib"
    supernet.save(model_file)


if __name__ == "__main__":
    main()
