"""
Unified Training & Evaluation Pipeline for Branch 1 (Deep Learning Ensemble).
Trains ResNet-50 and DenseNet-121 patch classifiers on H5 patches,
aggregates slide-level predictions, and fuses patient pN-stage distributions.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.branch1.patch_dataset import H5PatchDataset
from src.branch1.resnet_model import ResNetPatchClassifier
from src.branch1.densenet_model import DenseNetPatchClassifier
from src.branch1.patient_aggregator import PatientSlideAggregator
from src.branch1.stage_predictor import PatientStagePredictor
from src.branch1.ensemble_module import Branch1Ensemble


def train_patch_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str,
    epochs: int = 2,
    use_amp: bool = True,
):
    print(f"\n--- Training {model_name} on Patch Batches ---")
    model.train()
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"{model_name} Ep {ep}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                probs, logits, _ = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(labels)
            preds = torch.argmax(probs, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            acc = (correct / total) if total > 0 else 0.0
            pbar.set_postfix({"loss": f"{running_loss / total:.4f}", "acc": f"{acc * 100:.1f}%"})

    print(f"Finished training {model_name}. Final Patch Accuracy: {acc * 100:.2f}%")


@torch.no_grad()
def infer_slide_probabilities(
    model: nn.Module,
    h5_path: Path,
    device: torch.device,
    batch_size: int = 128,
    max_infer_patches: Optional[int] = None,
) -> np.ndarray:
    """
    Runs fast vectorized GPU model inference across patches of a slide.
    """
    import h5py

    model.eval()
    tumor_probs = []

    # ImageNet mean & std tensors on GPU
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with h5py.File(h5_path, "r") as f:
        images_ds = f["images"]
        n_patches = images_ds.shape[0]
        if max_infer_patches and n_patches > max_infer_patches:
            n_patches = max_infer_patches

        for i in range(0, n_patches, batch_size):
            end_idx = min(i + batch_size, n_patches)
            batch_imgs = images_ds[i:end_idx]  # (B, 224, 224, 3) uint8

            # Fast direct GPU upload & vectorized normalization
            tensors = torch.from_numpy(batch_imgs).permute(0, 3, 1, 2).to(device).float() / 255.0
            tensors = (tensors - mean) / std

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                probs, _, _ = model(tensors)

            tumor_probs.extend(probs[:, 1].cpu().float().numpy())

    return np.array(tumor_probs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate Branch 1 Ensemble")
    parser.add_argument("--h5_dir", type=str, default="output/branch1_patches")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_patches", type=int, default=500, help="Subset for fast iteration")
    parser.add_argument("--max_infer_patches", type=int, default=1000, help="Cap patches per slide during aggregation inference")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--pretrained", action="store_true", default=False, help="Download ImageNet weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    h5_dir = Path(args.h5_dir)
    slide_files = sorted(list(h5_dir.glob("*.h5")))
    print(f"Found {len(slide_files)} preprocessed slide archives in {h5_dir}")

    if len(slide_files) == 0:
        print("No H5 patch files found. Run run_preprocessing.py first.")
        return

    # Load patch dataset
    full_dataset = H5PatchDataset(h5_dir=h5_dir, is_training=True)
    n_patches = len(full_dataset)
    print(f"Total available patches: {n_patches}")

    # Subsample for training step if specified
    if args.max_train_patches and n_patches > args.max_train_patches:
        indices = np.random.choice(n_patches, args.max_train_patches, replace=False)
        train_ds = Subset(full_dataset, indices)
    else:
        train_ds = full_dataset

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Initialize models
    resnet = ResNetPatchClassifier(pretrained=args.pretrained).to(device)
    densenet = DenseNetPatchClassifier(pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()

    opt_resnet = torch.optim.Adam(resnet.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_densenet = torch.optim.Adam(densenet.parameters(), lr=args.lr, weight_decay=1e-4)

    # Train ResNet & DenseNet
    train_patch_model(resnet, train_loader, criterion, opt_resnet, device, "ResNet50", epochs=args.epochs)
    train_patch_model(densenet, train_loader, criterion, opt_densenet, device, "DenseNet121", epochs=args.epochs)

    # Slide & Patient Aggregation
    print("\n--- Running Slide-Level Aggregation & Patient pN Prediction ---")
    aggregator = PatientSlideAggregator()
    stage_predictor_resnet = PatientStagePredictor().to(device).eval()
    stage_predictor_densenet = PatientStagePredictor().to(device).eval()
    ensemble = Branch1Ensemble(weight_resnet=0.55, weight_densenet=0.45).to(device).eval()

    # Group slides by patient
    patient_slides_dict = {}
    for h5_p in slide_files:
        parts = h5_p.stem.split("_")
        pat_id = f"{parts[0]}_{parts[1]}"
        if pat_id not in patient_slides_dict:
            patient_slides_dict[pat_id] = []
        patient_slides_dict[pat_id].append(h5_p)

    output_results = []
    out_dir = Path("output/branch1_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pat_id, slides in patient_slides_dict.items():
        print(f"\nProcessing Patient: {pat_id} ({len(slides)} slides available)")

        resnet_slide_feats = []
        densenet_slide_feats = []

        for s_path in slides:
            # Inference with both models (fast GPU vectorized)
            p_resnet = infer_slide_probabilities(resnet, s_path, device, max_infer_patches=args.max_infer_patches)
            p_densenet = infer_slide_probabilities(densenet, s_path, device, max_infer_patches=args.max_infer_patches)

            sf_res = aggregator.extract_slide_features(p_resnet)
            sf_dense = aggregator.extract_slide_features(p_densenet)

            resnet_slide_feats.append(sf_res)
            densenet_slide_feats.append(sf_dense)

        # Assemble 25-D patient vectors
        vec_res = torch.from_numpy(aggregator.aggregate_patient_profile(resnet_slide_feats)).float().to(device)
        vec_dense = torch.from_numpy(aggregator.aggregate_patient_profile(densenet_slide_feats)).float().to(device)

        # Predict pN stage probabilities
        with torch.no_grad():
            stage_probs_res, _ = stage_predictor_resnet(vec_res)
            stage_probs_dense, _ = stage_predictor_densenet(vec_dense)
            fused_probs, info = ensemble(stage_probs_res, stage_probs_dense)

        res_record = {
            "patient_id": pat_id,
            "predicted_stage": info["predicted_stage"],
            "confidence": info["confidence"],
            "stage_probabilities": info["fused_distribution"],
            "resnet_probabilities": {name: float(stage_probs_res[0, i].item()) for i, name in enumerate(ensemble.STAGE_NAMES)},
            "densenet_probabilities": {name: float(stage_probs_dense[0, i].item()) for i, name in enumerate(ensemble.STAGE_NAMES)},
        }
        output_results.append(res_record)

        print(f"  Final Ensemble Prediction: {info['predicted_stage']} (Confidence: {info['confidence'] * 100:.1f}%)")
        print(f"  Distribution: {json.dumps(info['fused_distribution'], indent=2)}")

    # Save final Branch 1 predictions
    out_json = out_dir / "branch1_patient_predictions.json"
    with open(out_json, "w") as f:
        json.dump(output_results, f, indent=2)
    print(f"\nSaved Branch 1 predictions to: {out_json}")


if __name__ == "__main__":
    main()
