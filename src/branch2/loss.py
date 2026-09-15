"""
Hierarchical 3-Component Loss Function for Branch 2.
Implements Equations (9)-(12) from Tauqeer et al. (Scientific Reports 2025):
L_total = lambda_1 * L_instance + lambda_2 * L_attention + lambda_3 * L_bag
where:
- lambda_1 = 0.3 (Patch-level binary cross-entropy on instance predictions)
- lambda_2 = 0.2 (Attention regularization: L1 deviation from 1/K + gamma * L2 norm)
- lambda_3 = 0.5 (Slide-level weighted binary cross-entropy on bag prediction)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalBranch2Loss(nn.Module):
    """
    Combines instance-level patch supervision, selective attention regularization,
    and bag-level whole-slide classification loss.
    """

    def __init__(
        self,
        lambda_instance: float = 0.3,
        lambda_attention: float = 0.2,
        lambda_bag: float = 0.5,
        top_k: int = 4,
        gamma: float = 0.01,
        pos_weight_bag: float = 1.0,
        neg_weight_bag: float = 1.0,
    ):
        super().__init__()
        self.lambda_instance = lambda_instance
        self.lambda_attention = lambda_attention
        self.lambda_bag = lambda_bag
        self.top_k = top_k
        self.gamma = gamma
        self.pos_weight_bag = pos_weight_bag
        self.neg_weight_bag = neg_weight_bag

    def forward(
        self,
        patch_logits: torch.Tensor,
        patch_labels: torch.Tensor,
        attention_weights: torch.Tensor,
        slide_prob: torch.Tensor,
        slide_label: torch.Tensor,
        slide_logit: Optional[torch.Tensor] = None,
        patch_class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            patch_logits: (N, 2) raw patch classification logits
            patch_labels: (N,) ground-truth binary labels (0=normal, 1=tumor)
            attention_weights: (N, 8) or (N, 4) attention coefficients alpha
            slide_prob: Scalar in [0, 1] for slide prediction
            slide_label: Scalar 0 or 1 for slide ground truth
            slide_logit: Optional unnormalized slide logit for numerically stable BCE
            patch_class_weights: Optional (2,) class weights for imbalanced patch cross-entropy
            
        Returns:
            loss_total: Total scalar loss for backpropagation
            loss_dict: Dictionary containing broken-down loss metrics
        """
        # --- Component 1: Instance-Level Loss (Eq. 10) ---
        # Cross-entropy on patch predictions with class weighting
        l_instance = F.cross_entropy(patch_logits, patch_labels, weight=patch_class_weights)

        # --- Component 2: Attention Regularization Loss (Eq. 11) ---
        target_uniform = 1.0 / float(self.top_k)
        l1_term = torch.mean(torch.abs(attention_weights - target_uniform))
        l2_term = torch.mean(torch.sum(attention_weights ** 2, dim=-1))
        l_attention = l1_term + self.gamma * l2_term

        # --- Component 3: Bag-Level Classification Loss (Eq. 12) ---
        y_bag = slide_label.view(-1).float()
        if slide_logit is not None:
            # Numerically stable BCE with logits (prevents Inf/NaN in half precision)
            pos_weight = torch.tensor([self.pos_weight_bag], device=slide_logit.device, dtype=slide_logit.dtype)
            l_bag = F.binary_cross_entropy_with_logits(
                slide_logit.view(-1),
                y_bag.to(dtype=slide_logit.dtype),
                pos_weight=pos_weight,
            )
        else:
            eps = 1e-5
            p_bag = torch.clamp(slide_prob, eps, 1.0 - eps)
            l_bag = -(
                self.pos_weight_bag * y_bag * torch.log(p_bag)
                + self.neg_weight_bag * (1.0 - y_bag) * torch.log(1.0 - p_bag)
            ).mean()

        # Total Loss (Eq. 9)
        l_total = (
            self.lambda_instance * l_instance
            + self.lambda_attention * l_attention
            + self.lambda_bag * l_bag
        )

        loss_metrics = {
            "loss_total": float(l_total.item()),
            "loss_instance": float(l_instance.item()),
            "loss_attention": float(l_attention.item()),
            "loss_bag": float(l_bag.item()),
        }

        return l_total, loss_metrics
