"""
Slide-Level Classifier Head for Branch 2.
Implements Equations (7) & (8) from Tauqeer et al. (Scientific Reports 2025):
- Computes normalized attention weights beta_i for all N patches in a WSI
- Slide-level feature aggregation: S_slide = sum(beta_i * X_i)
- Slide-level binary metastasis prediction: Y_slide = sigmoid(Ws * S_slide)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideLevelClassifier(nn.Module):
    """
    Attention-based pooling module that aggregates N patch representations into a slide-level prediction.
    """

    def __init__(self, feature_dim: int = 1024, attn_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # Gated attention pooling network (Ilse et al. / Tauqeer et al.)
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

        # Slide-level classification layer (Ws)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_features: Tensor of shape (N, 1024) representing all N patches in a WSI.
            
        Returns:
            slide_prob: Scalar tensor in [0, 1] representing P(slide contains metastasis)
            slide_logit: Unnormalized scalar logit
            patch_weights: Tensor of shape (N, 1) containing attention weights beta_i
        """
        # Step 1: Compute raw attention scores for each patch -> (N, 1)
        raw_attn = self.attention_net(patch_features)

        # Step 2: Softmax over all N patches in the bag to obtain normalized beta_i -> (N, 1)
        patch_weights = F.softmax(raw_attn, dim=0)

        # Step 3: Weighted aggregation (Eq. 7): S_slide = sum(beta_i * X_i) -> (1, 1024)
        s_slide = torch.sum(patch_weights * patch_features, dim=0, keepdim=True)

        # Step 4: Slide-level classification (Eq. 8): Y_slide = sigmoid(Ws * S_slide)
        slide_logit = self.classifier(s_slide)  # (1, 1)
        slide_prob = torch.sigmoid(slide_logit)

        return slide_prob.squeeze(), slide_logit.squeeze(), patch_weights
