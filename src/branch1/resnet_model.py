"""
ResNet Backbone Module for Branch 1.
Implements ResNet50 patch-level tumor detection and slide-level representation pooling.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetPatchClassifier(nn.Module):
    """
    ResNet-50 patch classifier and feature extractor.
    Outputs:
    - Patch logits: (B, 2) [normal, tumor]
    - Patch features: (B, 2048)
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Feature extractor backbone (all layers up to avgpool)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        self.feature_dim = 2048

        # Patch classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes),
        )

        # Slide-level attention pooling head (Ilse et al.)
        self.attention_pool = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.slide_classifier = nn.Linear(self.feature_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for a batch of patches.
        
        Args:
            x: (B, 3, 224, 224)
            return_features: If True, returns 2048-D feature embeddings
            
        Returns:
            probs: (B, 2) Softmax probabilities
            logits: (B, 2) Raw unnormalized logits
            features: Optional (B, 2048) feature vectors
        """
        feats = self.feature_extractor(x)  # (B, 2048)
        logits = self.classifier(feats)    # (B, 2)
        probs = F.softmax(logits, dim=-1)

        return (probs, logits, feats) if return_features else (probs, logits, None)

    def pool_slide(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates N patch features of a slide into a slide metastasis prediction.
        
        Args:
            patch_features: (N, 2048)
            
        Returns:
            slide_prob: Scalar in [0, 1]
            slide_feat: (1, 2048) pooled slide feature representation
        """
        raw_attn = self.attention_pool(patch_features)  # (N, 1)
        weights = F.softmax(raw_attn, dim=0)            # (N, 1)
        slide_feat = torch.sum(weights * patch_features, dim=0, keepdim=True)  # (1, 2048)
        slide_prob = torch.sigmoid(self.slide_classifier(slide_feat)).squeeze()
        return slide_prob, slide_feat
