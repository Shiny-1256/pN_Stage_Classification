"""
Unified Training & Evaluation Pipeline for Branch 2.
Selective Neighborhood Attention & Automated pN-Staging.
Follows Tauqeer et al. (Scientific Reports 2025).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.branch2.neighborhood_builder import SpatialNeighborhoodBuilder
from src.branch2.selective_attention import SelectiveNeighborhoodAttention
from src.branch2.classifier_heads import SlideLevelClassifier
from src.branch2.loss import HierarchicalBranch2Loss
from src.branch2.staging_evaluator import TumorMapperAndStagingEvaluator


class Branch2SlideDataset(Dataset):
    """
    Dataset of preprocessed slides containing features, coordinates, and labels.
    """

    def __init__(self, feature_dir: str | Path, patch_size: int = 224):
        self.feature_dir = Path(feature_dir)
        self.slide_files = sorted(list(self.feature_dir.glob("*.pt")))
        self.patch_size = patch_size
        self.builder = SpatialNeighborhoodBuilder(patch_size=patch_size)

        # Quick pre-scan of slide-level labels for bag class weighting
        self.slide_labels = []
        for p in self.slide_files:
            data = torch.load(p, weights_only=True)
            has_tumor = int((data["labels"] == 1).any())
            self.slide_labels.append(has_tumor)

    def __len__(self) -> int:
        return len(self.slide_files)

    def __getitem__(self, idx: int) -> dict:
        pt_path = self.slide_files[idx]
        data = torch.load(pt_path, weights_only=True)

        features = data["features"]  # (N, D)
        coords = data["coords"]      # (N, 2)
        labels = data["labels"]      # (N,)

        # Build 8-neighbor spatial adjacency graph
        neighbor_indices, neighbor_mask = self.builder.build_neighborhood_indices(coords)

        # Slide-level binary label (1 if any patch is tumor, else 0)
        slide_label = self.slide_labels[idx]

        return {
            "slide_id": data.get("slide_id", pt_path.stem),
            "features": features,
            "coords": coords,
            "labels": labels,
            "neighbor_indices": neighbor_indices,
            "neighbor_mask": neighbor_mask,
            "slide_label": torch.tensor(slide_label, dtype=torch.float32),
        }


class Branch2Model(nn.Module):
    """
    Unified Branch 2 model combining Selective Neighborhood Attention (patch-level)
    and attention-based pooling (slide-level).
    """

    def __init__(self, in_features: int = 1024, attn_dim: int = 256, top_k: int = 4):
        super().__init__()
        # If input features differ from 1024, project to 1024
        self.in_proj = nn.Identity() if in_features == 1024 else nn.Linear(in_features, 1024)

        self.sna = SelectiveNeighborhoodAttention(
            feature_dim=1024,
            attn_dim=attn_dim,
            top_k=top_k,
            num_classes=2,
        )
        self.slide_classifier = SlideLevelClassifier(feature_dim=1024, attn_dim=attn_dim)

    def forward(
        self,
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
        chunk_size: int = 2048,
    ):
        """
        Args:
            features: (N, D) patch representations
            neighbor_indices: (N, 8) indices of adjacent patches
            neighbor_mask: (N, 8) validity mask
            chunk_size: Maximum patches to process in a single GPU pass to respect 6GB VRAM.
        """
        N = features.shape[0]
        feats_1024 = self.in_proj(features)

        # If N is small enough, process directly
        if N <= chunk_size:
            neighbor_feats = feats_1024[neighbor_indices]
            patch_probs, patch_logits, alpha, context_vec = self.sna(
                center_feats=feats_1024,
                neighbor_feats=neighbor_feats,
                neighbor_mask=neighbor_mask,
            )
        else:
            # Process in memory-safe chunks of chunk_size
            probs_list, logits_list, alpha_list, context_list = [], [], [], []
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                c_feats = feats_1024[start:end]
                n_indices_chunk = neighbor_indices[start:end]
                n_mask_chunk = neighbor_mask[start:end]
                n_feats_chunk = feats_1024[n_indices_chunk]

                p_prob, p_logit, p_alpha, p_ctx = self.sna(
                    center_feats=c_feats,
                    neighbor_feats=n_feats_chunk,
                    neighbor_mask=n_mask_chunk,
                )
                probs_list.append(p_prob)
                logits_list.append(p_logit)
                alpha_list.append(p_alpha)
                context_list.append(p_ctx)

            patch_probs = torch.cat(probs_list, dim=0)
            patch_logits = torch.cat(logits_list, dim=0)
            alpha = torch.cat(alpha_list, dim=0)
            context_vec = torch.cat(context_list, dim=0)

        # Slide-level Attention Pooling
        slide_prob, slide_logit, patch_weights = self.slide_classifier(context_vec)

        return {
            "patch_probs": patch_probs,
            "patch_logits": patch_logits,
            "alpha": alpha,
            "context_vec": context_vec,
            "slide_prob": slide_prob,
            "slide_logit": slide_logit,
            "patch_weights": patch_weights,
        }


def train_epoch(
    model: nn.Module,
    dataset: Branch2SlideDataset,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    max_patches_per_slide: int = 2048,
    patch_class_weights: torch.Tensor = None,
) -> dict:
    model.train()
    total_loss = 0.0
    metrics_sum = {"loss_instance": 0.0, "loss_attention": 0.0, "loss_bag": 0.0}
    valid_steps = 0

    # Select hardware-optimized AMP precision (bfloat16 has full float32 dynamic range, avoiding underflows)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Interleave negative and positive slides so positive tumor gradients are not erased
    pos_slide_indices = [i for i, lbl in enumerate(dataset.slide_labels) if lbl == 1]
    neg_slide_indices = [i for i, lbl in enumerate(dataset.slide_labels) if lbl == 0]

    if len(pos_slide_indices) > 0:
        neg_perm = np.random.permutation(neg_slide_indices)
        pos_sampled = np.random.choice(pos_slide_indices, size=len(neg_slide_indices), replace=True)
        epoch_indices = []
        for n_idx, p_idx in zip(neg_perm, pos_sampled):
            epoch_indices.extend([n_idx, p_idx])
    else:
        epoch_indices = list(range(len(dataset)))

    for idx in epoch_indices:
        sample = dataset[int(idx)]
        features = sample["features"]
        labels = sample["labels"]
        slide_label = sample["slide_label"].to(device)

        pos_idx = torch.where(labels == 1)[0]
        neg_idx = torch.where(labels == 0)[0]

        if len(pos_idx) > 0:
            # Positive slide: balanced 1:1 sampling (up to 256 tumor + 256 normal)
            n_pos = min(len(pos_idx), 256)
            s_pos = pos_idx[torch.randperm(len(pos_idx))[:n_pos]]
            s_neg = neg_idx[torch.randperm(len(neg_idx))[:n_pos]]
            sub_idx = torch.cat([s_pos, s_neg])
        else:
            # Normal slide: 256 patches
            n_sample = min(len(neg_idx), 256)
            sub_idx = neg_idx[torch.randperm(len(neg_idx))[:n_sample]]

        features = features[sub_idx].to(device)
        labels = labels[sub_idx].to(device)
        neighbor_indices = torch.arange(len(sub_idx), device=device).unsqueeze(1).expand(-1, 8)
        neighbor_mask = torch.ones((len(sub_idx), 8), dtype=torch.bool, device=device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(features, neighbor_indices, neighbor_mask)
            loss, loss_dict = criterion(
                patch_logits=out["patch_logits"],
                patch_labels=labels,
                attention_weights=out["alpha"],
                slide_prob=out["slide_prob"],
                slide_label=slide_label,
                slide_logit=out["slide_logit"],
                patch_class_weights=patch_class_weights,
            )

        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss on slide {sample['slide_id']}, skipping step.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in metrics_sum:
            metrics_sum[k] += loss_dict.get(k, 0.0)
        valid_steps += 1

        if device.type == "cuda":
            torch.cuda.empty_cache()

    denom = max(1, valid_steps)
    return {
        "loss": total_loss / denom,
        "loss_instance": metrics_sum["loss_instance"] / denom,
        "loss_attention": metrics_sum["loss_attention"] / denom,
        "loss_bag": metrics_sum["loss_bag"] / denom,
    }


def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate Branch 2")
    parser.add_argument("--features_dir", type=str, default="output/branch2_features")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tumor_threshold", type=float, default=0.60)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    dataset = Branch2SlideDataset(args.features_dir)
    print(f"Found {len(dataset)} preprocessed slides in {args.features_dir}")

    if len(dataset) == 0:
        print("No preprocessed slides found. Please run preprocessing first.")
        return

    # Calculate dataset class balance
    num_pos_slides = sum(dataset.slide_labels)
    num_neg_slides = len(dataset) - num_pos_slides
    print(f"Dataset Distribution: {num_neg_slides} negative slides, {num_pos_slides} positive slides (balanced alternating schedule)")

    # Infer feature dimension from first slide
    sample_feat = dataset[0]["features"]
    in_dim = sample_feat.shape[1]
    print(f"Input feature dimension: {in_dim}")

    model = Branch2Model(in_features=in_dim, attn_dim=256, top_k=4).to(device)
    criterion = HierarchicalBranch2Loss(pos_weight_bag=1.0, lambda_instance=0.7, lambda_bag=0.2, lambda_attention=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    patch_class_weights = torch.tensor([1.0, 3.0], device=device)

    print("\nStarting Branch 2 training...")
    for ep in range(1, args.epochs + 1):
        metrics = train_epoch(
            model=model,
            dataset=dataset,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            patch_class_weights=patch_class_weights,
        )
        print(f"Epoch [{ep}/{args.epochs}] -> Total Loss: {metrics['loss']:.4f} | "
              f"Instance Loss: {metrics['loss_instance']:.4f} | "
              f"Attention Reg: {metrics['loss_attention']:.4f} | "
              f"Bag Loss: {metrics['loss_bag']:.4f}")

    # Save trained model weights
    weights_dir = PROJECT_ROOT / "output" / "branch2_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "branch2_model.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"\nSaved trained Branch 2 model weights to: {weights_path}")

    # Run inference and demonstrate staging
    print("\nRunning Staging Evaluation across processed slides...")
    evaluator = TumorMapperAndStagingEvaluator()
    model.eval()

    slide_summaries = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            features = sample["features"].to(device)
            coords = sample["coords"].numpy()
            neighbor_indices = sample["neighbor_indices"].to(device)
            neighbor_mask = sample["neighbor_mask"].to(device)

            out = model(features, neighbor_indices, neighbor_mask, chunk_size=2048)
            tumor_probs = out["patch_probs"][:, 1].cpu().numpy()

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Reconstruct tumor map with configurable probability threshold
            bmap, hmap = evaluator.reconstruct_tumor_map(
                coords=coords,
                predictions=tumor_probs,
                wsi_width=int(coords[:, 0].max() + 256),
                wsi_height=int(coords[:, 1].max() + 256),
                downsample_factor=16,
                threshold=args.tumor_threshold,
            )

            # Quantify metastases
            summary = evaluator.quantify_slide_metastases(bmap, downsample_factor=16)
            summary["slide_id"] = sample["slide_id"]
            slide_summaries.append(summary)

            print(f"Slide: {sample['slide_id']} -> Highest Category: {summary['highest_category']} "
                  f"(Macro: {summary['num_macro']}, Micro: {summary['num_micro']}, ITC: {summary['num_itc']})")

    # Load ground-truth labels if available
    stage_csv = PROJECT_ROOT.parent / "dataset" / "stage_labels.csv"
    gt_patient_stages = {}
    gt_slide_stages = {}
    if stage_csv.exists():
        df_stages = pd.read_csv(stage_csv)
        for _, row in df_stages.iterrows():
            name = str(row["patient"])
            stage = str(row["stage"]).strip()
            if name.endswith(".zip"):
                p_id = name.replace(".zip", "")
                gt_patient_stages[p_id] = stage
            elif name.endswith(".tif"):
                s_id = name.replace(".tif", "")
                gt_slide_stages[s_id] = stage

    # Group slides by patient ID
    patient_slides = defaultdict(list)
    for s in slide_summaries:
        patient_id = "_".join(s["slide_id"].split("_")[:2])
        patient_slides[patient_id].append(s)

    patient_predictions = {}

    print("\n" + "=" * 70)
    print("PATIENT-LEVEL pN-STAGE EVALUATION (Branch 2):")
    print("=" * 70)
    for patient_id, summaries in sorted(patient_slides.items()):
        predicted_stage = evaluator.compute_patient_pn_stage(summaries)
        gt_stage = gt_patient_stages.get(patient_id, "Unknown")
        p_macro = sum(s["num_macro"] for s in summaries)
        p_micro = sum(s["num_micro"] for s in summaries)
        p_itc = sum(s["num_itc"] for s in summaries)
        pos_nodes = sum(1 for s in summaries if s["highest_category"] in ["micro", "macro"])

        patient_predictions[patient_id] = {
            "patient_id": patient_id,
            "predicted_pn_stage": predicted_stage,
            "ground_truth_pn_stage": gt_stage,
            "num_nodes_examined": len(summaries),
            "positive_nodes": pos_nodes,
            "num_macro": p_macro,
            "num_micro": p_micro,
            "num_itc": p_itc,
            "slide_breakdown": [
                {
                    "slide_id": s["slide_id"],
                    "highest_category": s["highest_category"],
                    "num_macro": s["num_macro"],
                    "num_micro": s["num_micro"],
                    "num_itc": s["num_itc"],
                    "ground_truth": gt_slide_stages.get(s["slide_id"], "Unknown"),
                }
                for s in summaries
            ],
        }

        match_str = "[MATCH]" if predicted_stage.lower() == gt_stage.lower() else "[DISCREPANCY]"
        print(f"Patient: {patient_id} ({len(summaries)} lymph nodes)")
        print(f"  • Estimated pN-Stage : {predicted_stage}")
        print(f"  • Ground Truth Stage : {gt_stage}  {match_str}")
        print(f"  • Positive Nodes     : {pos_nodes}/{len(summaries)} with metastasis")
        print(f"  • Lesions Detected   : Macro={p_macro}, Micro={p_micro}, ITC={p_itc}")
        print("-" * 70)

    # Export predictions to JSON for Late Fusion Module
    out_dir = PROJECT_ROOT / "output" / "branch2_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "branch2_patient_predictions.json"
    with open(out_path, "w") as f:
        json.dump(patient_predictions, f, indent=2)
    print(f"\nSaved patient-level Branch 2 predictions to: {out_path}")


if __name__ == "__main__":
    main()
