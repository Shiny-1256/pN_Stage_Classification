"""
Nuclei Feature Extractor (NFE) for Branch 2.
Constructs a 16-channel cellular feature map representing:
- 5 channels: Cell type probabilities (Neoplastic, Inflammatory, Connective, Epithelial, Other)
- 7 channels: Hu moment invariants (nuclear morphology, elongation, asymmetry)
- 2 channels: Mean & std of RGB intensities inside nuclei (chromatin density)
- 2 channels: Shannon entropy & Local Binary Patterns (LBP) texture (chromatin heterogeneity)
Downsampled via Average Pooling (256x256 -> 64x64) and compressed via MLP (16 -> 1 channel).
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.feature import local_binary_pattern


class NucleiMLPCompressor(nn.Module):
    """
    MLP / Conv block to compress 16-channel nuclei feature maps (64x64x16)
    into a single 64x64x1 feature map.
    """

    def __init__(self, in_channels: int = 16, hidden_dim: int = 32, out_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 16, 64, 64)
        Returns:
            Tensor of shape (B, 1, 64, 64)
        """
        return self.net(x)


class NucleiFeatureExtractor(nn.Module):
    """
    End-to-end Nuclei Feature Extractor module.
    Takes 16-channel input or extracts 16 channels from RGB patch & nuclei mask,
    downsamples to 64x64, and compresses to 64x64x1.
    """

    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((64, 64))
        self.compressor = NucleiMLPCompressor(in_channels=in_channels, out_channels=1)

    @staticmethod
    def extract_16ch_features_from_patch(
        patch_rgb: np.ndarray,
        nuclei_mask: Optional[np.ndarray] = None,
        cell_type_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Constructs the 16-channel nuclei feature array for a single patch (256, 256, 16).
        
        Args:
            patch_rgb: (256, 256, 3) uint8 RGB image
            nuclei_mask: Optional (256, 256) binary mask (1=nucleus, 0=background)
            cell_type_probs: Optional (256, 256, 5) float array of 5 cell-type probabilities
            
        Returns:
            feature_16ch: (256, 256, 16) float32 array
        """
        h, w = patch_rgb.shape[:2]
        feat_16ch = np.zeros((h, w, 16), dtype=np.float32)

        # If no external nuclei mask is provided, perform hematoxylin thresholding
        if nuclei_mask is None:
            # Hematoxylin separates in optical density: Blue-Purple stain
            gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
            # Otsu threshold on inverted grayscale (nuclei are darker than background)
            inv = 255 - gray
            _, nuclei_mask = cv2.threshold(inv, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Remove very tiny specs (< 5 pixels)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel)

        has_nuclei = np.count_nonzero(nuclei_mask) > 0

        # --- Channels 0-4: 5 Cell Type Probabilities ---
        if cell_type_probs is not None and cell_type_probs.shape == (h, w, 5):
            feat_16ch[:, :, 0:5] = cell_type_probs
        else:
            # Baseline estimation from color/morphology:
            # Neoplastic (high N:C ratio, dark, large), Inflammatory (small, dense),
            # Connective (elongated, faint), Epithelial, Other
            if has_nuclei:
                # Approximate baseline distribution on nuclei pixels
                n_mask_bool = nuclei_mask.astype(bool)
                feat_16ch[n_mask_bool, 0] = 0.40  # neoplastic default prior
                feat_16ch[n_mask_bool, 1] = 0.20  # inflammatory
                feat_16ch[n_mask_bool, 2] = 0.20  # connective
                feat_16ch[n_mask_bool, 3] = 0.15  # epithelial
                feat_16ch[n_mask_bool, 4] = 0.05  # other

        # --- Channels 5-11: 7 Hu Moment Invariants (Shape) ---
        if has_nuclei:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                nuclei_mask.astype(np.uint8), connectivity=8
            )
            for lbl in range(1, num_labels):
                comp_mask = (labels == lbl).astype(np.uint8)
                moments = cv2.moments(comp_mask)
                hu = cv2.HuMoments(moments).flatten()
                # Log-scale Hu moments to stabilize dynamic range
                hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
                for ch in range(7):
                    feat_16ch[labels == lbl, 5 + ch] = hu_log[ch]

        # --- Channels 12-13: Mean & Std RGB within Nuclei (Chromatin Density) ---
        if has_nuclei:
            n_mask_bool = nuclei_mask.astype(bool)
            nuclei_pixels = patch_rgb[n_mask_bool].astype(np.float32)
            mean_intensity = float(np.mean(nuclei_pixels)) / 255.0
            std_intensity = float(np.std(nuclei_pixels)) / 255.0
            feat_16ch[n_mask_bool, 12] = mean_intensity
            feat_16ch[n_mask_bool, 13] = std_intensity

        # --- Channels 14-15: Entropy & LBP Texture ---
        gray_f = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
        # Shannon entropy via local 9x9 window
        # Approximation: local variance as proxy for local entropy / complexity
        local_mean = cv2.blur(gray_f.astype(np.float32), (9, 9))
        local_sq_mean = cv2.blur(gray_f.astype(np.float32) ** 2, (9, 9))
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
        local_entropy = np.log1p(local_var) / 10.0  # Normalized to ~[0, 1]

        # Local Binary Patterns (P=8, R=1)
        lbp = local_binary_pattern(gray_f, P=8, R=1, method="uniform") / 10.0

        if has_nuclei:
            feat_16ch[n_mask_bool, 14] = local_entropy[n_mask_bool]
            feat_16ch[n_mask_bool, 15] = lbp[n_mask_bool]
        else:
            feat_16ch[:, :, 14] = local_entropy * 0.1
            feat_16ch[:, :, 15] = lbp * 0.1

        return feat_16ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 16, H, W) where (H, W) is typically 256x256 or 64x64.
        Returns:
            Tensor of shape (B, 1, 64, 64)
        """
        # Step 1: Average pooling down to (64, 64) if needed
        if x.shape[-2:] != (64, 64):
            x = self.avg_pool(x)
        # Step 2: MLP compression from 16 channels to 1 channel
        out = self.compressor(x)
        return out
