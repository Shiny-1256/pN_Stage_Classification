"""
Dual-Path Feature Fusion Module for Branch 2.
Combines Nuclei Feature Map (64x64x1) and Tissue Feature Map (64x64x1):
- Concatenation: 64x64x2
- Flatten: 8,192 elements
- Linear transformation: Wconcat (8192 -> 1024)
Yields the unified 1024-D patch feature representation F_(i,j).
"""

import torch
import torch.nn as nn


class DualPathFeatureFusion(nn.Module):
    """
    Fuses nuclei-level and tissue-level feature maps into a single 1024-D vector.
    """

    def __init__(self, in_channels: int = 2, spatial_dim: int = 64, out_dim: int = 1024):
        super().__init__()
        flat_dim = in_channels * spatial_dim * spatial_dim  # 2 * 64 * 64 = 8192
        self.w_concat = nn.Linear(flat_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, nuclei_feat: torch.Tensor, tissue_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nuclei_feat: (B, 1, 64, 64)
            tissue_feat: (B, 1, 64, 64)
            
        Returns:
            f_ij: (B, 1024) fused patch feature representations
        """
        # Step 1: Concatenate along channel axis -> (B, 2, 64, 64)
        fused_map = torch.cat([nuclei_feat, tissue_feat], dim=1)

        # Step 2: Flatten to 8192-D vector -> (B, 8192)
        flat = fused_map.flatten(start_dim=1)

        # Step 3: Linear transformation with Wconcat -> (B, 1024)
        f_ij = self.w_concat(flat)
        f_ij = self.layer_norm(f_ij)

        return f_ij
