"""
Cleans up all artificial/replicated patient data from the output directories.
Restores the dataset strictly to the 4 genuine CAMELYON17 patients:
- patient_000
- patient_001
- patient_004
- patient_015
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GENUINE_PATIENTS = {"patient_000", "patient_001", "patient_004", "patient_015"}


def clean_visualizations():
    vis_dir = PROJECT_ROOT / "output" / "slide_visualizations"
    mask_dir = PROJECT_ROOT / "output" / "tissue_masks"

    deleted_vis = 0
    if vis_dir.exists():
        for f in vis_dir.glob("patient_*.png"):
            pid = "_".join(f.stem.split("_")[:2])
            if pid not in GENUINE_PATIENTS:
                f.unlink()
                deleted_vis += 1

    deleted_masks = 0
    if mask_dir.exists():
        for f in mask_dir.glob("patient_*.png"):
            pid = "_".join(f.stem.split("_")[:2])
            if pid not in GENUINE_PATIENTS:
                f.unlink()
                deleted_masks += 1

    # Filter slide_meta.json
    meta_file = vis_dir / "slide_meta.json"
    if meta_file.exists():
        with open(meta_file, "r") as fp:
            meta = json.load(fp)
        filtered_meta = {
            k: v for k, v in meta.items()
            if "_".join(k.split("_")[:2]) in GENUINE_PATIENTS
        }
        with open(meta_file, "w") as fp:
            json.dump(filtered_meta, fp, indent=2)
        print(f"Filtered slide_meta.json: {len(filtered_meta)} slides remaining.")

    print(f"Purged {deleted_vis} fake visual files and {deleted_masks} fake mask files.")


def clean_predictions():
    # Filter branch1 predictions
    b1_file = PROJECT_ROOT / "output" / "branch1_predictions" / "branch1_patient_predictions.json"
    if b1_file.exists():
        with open(b1_file, "r") as fp:
            b1_data = json.load(fp)
        filtered_b1 = [item for item in b1_data if item.get("patient_id") in GENUINE_PATIENTS]
        with open(b1_file, "w") as fp:
            json.dump(filtered_b1, fp, indent=2)
        print(f"Filtered branch1_patient_predictions.json: {len(filtered_b1)} patients remaining.")

    # Filter branch2 predictions
    b2_file = PROJECT_ROOT / "output" / "branch2_predictions" / "branch2_patient_predictions.json"
    if b2_file.exists():
        with open(b2_file, "r") as fp:
            b2_data = json.load(fp)
        filtered_b2 = {k: v for k, v in b2_data.items() if k in GENUINE_PATIENTS}
        with open(b2_file, "w") as fp:
            json.dump(filtered_b2, fp, indent=2)
        print(f"Filtered branch2_patient_predictions.json: {len(filtered_b2)} patients remaining.")


if __name__ == "__main__":
    clean_visualizations()
    clean_predictions()
    print("Data purge complete. Only genuine CAMELYON17 patients retained.")
