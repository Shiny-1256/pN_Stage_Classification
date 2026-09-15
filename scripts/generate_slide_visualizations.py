"""
Precomputes and exports tumor heatmaps, binary detection masks, and visualization overlays
for all 20 lymph node slides using trained Branch 2 model.
"""

import sys
import json
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_branch2 import Branch2SlideDataset, Branch2Model, TumorMapperAndStagingEvaluator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating slide visual assets using device: {device}")

    feat_dir = PROJECT_ROOT / "output" / "branch2_features"
    mask_dir = PROJECT_ROOT / "output" / "tissue_masks"
    vis_dir = PROJECT_ROOT / "output" / "slide_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    dataset = Branch2SlideDataset(feat_dir, patch_size=256)
    print(f"Loaded {len(dataset)} slides.")

    # Initialize model
    model = Branch2Model(in_features=2048, attn_dim=256, top_k=4).to(device)
    weights_path = PROJECT_ROOT / "output" / "branch2_weights" / "branch2_model.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded trained Branch 2 weights from: {weights_path}")
    else:
        print("Warning: Model weights not found. Using initialized weights.")

    model.eval()
    evaluator = TumorMapperAndStagingEvaluator(pixel_size_um=0.24, patch_size_px=256)

    slide_meta = {}

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Generating Heatmaps"):
            sample = dataset[idx]
            slide_id = sample["slide_id"]
            features = sample["features"].to(device)
            coords = sample["coords"].numpy()
            neighbor_indices = sample["neighbor_indices"].to(device)
            neighbor_mask = sample["neighbor_mask"].to(device)

            out = model(features, neighbor_indices, neighbor_mask, chunk_size=2048)
            patch_probs = out["patch_probs"][:, 1].cpu().numpy()

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Downsample factor 32 for smooth thumbnail-scale visualization
            ds_factor = 32
            wsi_w = int(coords[:, 0].max() + 256)
            wsi_h = int(coords[:, 1].max() + 256)

            bmap, hmap = evaluator.reconstruct_tumor_map(
                coords=coords,
                predictions=patch_probs,
                wsi_width=wsi_w,
                wsi_height=wsi_h,
                downsample_factor=ds_factor,
                threshold=0.5,
            )

            # Convert continuous heatmap to RGB colormap (Jet)
            hmap_norm = np.clip(hmap, 0.0, 1.0)
            hmap_u8 = (hmap_norm * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)

            # Load tissue mask if available
            mask_path = mask_dir / f"{slide_id}_mask.png"
            if mask_path.exists():
                mask_img = cv2.imread(str(mask_path))
                # Resize heatmap to match mask dimensions exactly
                h_target, w_target = mask_img.shape[:2]
                heatmap_resized = cv2.resize(heatmap_color, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
                hmap_norm_resized = cv2.resize(hmap_norm, (w_target, h_target), interpolation=cv2.INTER_LINEAR)

                # Alpha blend where probability > 0.05
                alpha_mask = np.repeat((hmap_norm_resized > 0.05)[:, :, np.newaxis], 3, axis=2)
                blended = mask_img.copy()
                blended[alpha_mask] = cv2.addWeighted(
                    mask_img, 0.45, heatmap_resized, 0.55, 0
                )[alpha_mask]
            else:
                heatmap_resized = heatmap_color
                blended = heatmap_color

            # Save visualization artifacts
            cv2.imwrite(str(vis_dir / f"{slide_id}_heatmap.png"), heatmap_resized)
            cv2.imwrite(str(vis_dir / f"{slide_id}_overlay.png"), blended)
            cv2.imwrite(str(vis_dir / f"{slide_id}_binary.png"), (bmap * 255).astype(np.uint8))

            # Metastasis quantification summary
            quant = evaluator.quantify_slide_metastases(bmap, downsample_factor=ds_factor)

            slide_meta[slide_id] = {
                "slide_id": slide_id,
                "num_patches": int(len(coords)),
                "max_prob": float(np.max(patch_probs)),
                "mean_prob": float(np.mean(patch_probs)),
                "p95_prob": float(np.percentile(patch_probs, 95)),
                "highest_category": quant["highest_category"],
                "num_macro": int(quant["num_macro"]),
                "num_micro": int(quant["num_micro"]),
                "num_itc": int(quant["num_itc"]),
                "wsi_width": wsi_w,
                "wsi_height": wsi_h,
            }

    with open(vis_dir / "slide_meta.json", "w") as f:
        json.dump(slide_meta, f, indent=2)

    print(f"\nAll slide visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
