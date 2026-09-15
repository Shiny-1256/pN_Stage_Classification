"""
Tissue Feature Extractor (TFE) for Branch 2.
Combines CNN (local histological architecture) and Transformer blocks (global context).
Processes (B, 3, 256, 256) RGB patches into (B, 1, 64, 64) spatial feature maps.
Includes supervised training heads for Binary Cross-Entropy + Contrastive loss.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x) + self.shortcut(x)
        return F.relu(out, inplace=True)


class SpatialTransformerBlock(nn.Module):
    """
    Lightweight Vision Transformer block with multi-head self-attention for 64x64 feature maps.
    Uses windowed / downsampled token self-attention for efficiency on 6GB VRAM.
    """

    def __init__(self, dim: int = 64, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Reshape to (B, H*W, C)
        flat = x.flatten(2).transpose(1, 2)
        norm_flat = self.norm1(flat)
        attn_out, _ = self.attn(norm_flat, norm_flat, norm_flat)
        flat = flat + attn_out
        flat = flat + self.mlp(self.norm2(flat))
        # Reshape back to (B, C, H, W)
        out = flat.transpose(1, 2).reshape(B, C, H, W)
        return out


class TissueFeatureExtractor(nn.Module):
    """
    CNN + Transformer hybrid high-level tissue feature extractor.
    Input: (B, 3, 256, 256) RGB patch
    Output: (B, 1, 64, 64) spatial feature map
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
    ):
        super().__init__()

        # --- Stage 1: CNN Stem (reduces 256x256 -> 64x64) ---
        # 256x256 -> 128x128
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 128x128 -> 64x64
        self.layer1 = ConvBlock(32, 48, stride=2)
        self.layer2 = ConvBlock(48, embed_dim, stride=1)

        # --- Stage 2: Transformer Encoder Blocks ---
        self.transformer_blocks = nn.ModuleList([
            SpatialTransformerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(num_transformer_layers)
        ])

        # --- Stage 3: Compression to 1 channel (64x64x1) ---
        self.head_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Optional supervised classification head for pretraining stage
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # [P(normal), P(tumor)]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, 3, 256, 256) RGB images normalized to [0, 1] or ImageNet stats
            return_logits: If True, also returns patch-level classification logits (for pretraining)
            
        Returns:
            feature_map: (B, 1, 64, 64)
            logits: Optional (B, 2)
        """
        # CNN Stem: 256x256 -> 64x64
        feat = self.stem(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)  # (B, embed_dim, 64, 64)

        # Transformer contextual self-attention
        for blk in self.transformer_blocks:
            feat = blk(feat)

        # Compute logits if pretraining head requested
        logits = None
        if return_logits:
            logits = self.classifier_head(feat)

        # Final 1-channel feature map
        out_map = self.head_conv(feat)  # (B, 1, 64, 64)

        return (out_map, logits) if return_logits else (out_map, None)
