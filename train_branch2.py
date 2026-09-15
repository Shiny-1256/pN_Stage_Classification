"""
Unified Training & Evaluation Pipeline for Branch 2.
Selective Neighborhood Attention & Automated pN-Staging.
Follows Tauqeer et al. (Scientific Reports 2025).
"""

import argparse
import sys
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
        slide_label = 1 if (labels == 1).any() else 0

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
    ):
        """
        Args:
            features: (N, D) patch representations
            neighbor_indices: (N, 8) indices of adjacent patches
            neighbor_mask: (N, 8) validity mask
        """
        feats_1024 = self.in_proj(features)

        # Gather 8 neighbor features for every patch: (N, 8, 1024)
        neighbor_feats = feats_1024[neighbor_indices]

        # Patch-level Selective Neighborhood Attention
        patch_probs, patch_logits, alpha, context_vec = self.sna(
            center_feats=feats_1024,
            neighbor_feats=neighbor_feats,
            neighbor_mask=neighbor_mask,
        )

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
) -> dict:
    model.train()
    total_loss = 0.0
    metrics_sum = {"loss_instance": 0.0, "loss_attention": 0.0, "loss_bag": 0.0}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        features = sample["features"].to(device)
        labels = sample["labels"].to(device)
        neighbor_indices = sample["neighbor_indices"].to(device)
        neighbor_mask = sample["neighbor_mask"].to(device)
        slide_label = sample["slide_label"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(features, neighbor_indices, neighbor_mask)
            loss, loss_dict = criterion(
                patch_logits=out["patch_logits"],
                patch_labels=labels,
                attention_weights=out["alpha"],
                slide_prob=out["slide_prob"],
                slide_label=slide_label,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in metrics_sum:
            metrics_sum[k] += loss_dict.get(k, 0.0)

    num_samples = max(1, len(dataset))
    return {
        "loss": total_loss / num_samples,
        "loss_instance": metrics_sum["loss_instance"] / num_samples,
        "loss_attention": metrics_sum["loss_attention"] / num_samples,
        "loss_bag": metrics_sum["loss_bag"] / num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate Branch 2")
    parser.add_argument("--features_dir", type=str, default="output/branch2_features")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    dataset = Branch2SlideDataset(args.features_dir)
    print(f"Found {len(dataset)} preprocessed slides in {args.features_dir}")

    if len(dataset) == 0:
        print("No preprocessed slides found. Please run preprocessing first.")
        return

    # Infer feature dimension from first slide
    sample_feat = dataset[0]["features"]
    in_dim = sample_feat.shape[1]
    print(f"Input feature dimension: {in_dim}")

    model = Branch2Model(in_features=in_dim, attn_dim=256, top_k=4).to(device)
    criterion = HierarchicalBranch2Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    print("\nStarting Branch 2 training...")
    for ep in range(1, args.epochs + 1):
        metrics = train_epoch(model, dataset, criterion, optimizer, device)
        print(f"Epoch [{ep}/{args.epochs}] -> Total Loss: {metrics['loss']:.4f} | "
              f"Instance Loss: {metrics['loss_instance']:.4f} | "
              f"Attention Reg: {metrics['loss_attention']:.4f} | "
              f"Bag Loss: {metrics['loss_bag']:.4f}")

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

            out = model(features, neighbor_indices, neighbor_mask)
            tumor_probs = out["patch_probs"][:, 1].cpu().numpy()

            # Reconstruct tumor map
            bmap, hmap = evaluator.reconstruct_tumor_map(
                coords=coords,
                predictions=tumor_probs,
                wsi_width=int(coords[:, 0].max() + 256),
                wsi_height=int(coords[:, 1].max() + 256),
                downsample_factor=16,
            )

            # Quantify metastases
            summary = evaluator.quantify_slide_metastases(bmap, downsample_factor=16)
            summary["slide_id"] = sample["slide_id"]
            slide_summaries.append(summary)

            print(f"Slide: {sample['slide_id']} -> Highest Category: {summary['highest_category']} "
                  f"(Micro: {summary['num_micro']}, Macro: {summary['num_macro']}, ITC: {summary['num_itc']})")

    # Staging
    patient_pn_stage = evaluator.compute_patient_pn_stage(slide_summaries)
    print(f"\n=======================================================")
    print(f"Patient pN-Stage Estimation: {patient_pn_stage}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
