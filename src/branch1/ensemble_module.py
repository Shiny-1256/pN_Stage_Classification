"""
Deep Learning Ensemble Module for Branch 1.
Combines ResNet and DenseNet predictions via weighted probability blending
or learnable stacking to produce the final Branch 1 pN-stage distribution.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Branch1Ensemble(nn.Module):
    """
    Ensemble module that fuses ResNet and DenseNet patient-level pN stage predictions.
    Supports fixed weighted averaging and learnable gating.
    """

    STAGE_NAMES = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]

    def __init__(
        self,
        weight_resnet: float = 0.50,
        weight_densenet: float = 0.50,
        use_learnable_gate: bool = False,
        num_stages: int = 5,
    ):
        super().__init__()
        self.use_learnable_gate = use_learnable_gate
        self.num_stages = num_stages

        if use_learnable_gate:
            # Gating network that dynamically adjusts weights based on model predictions
            self.gate = nn.Sequential(
                nn.Linear(num_stages * 2, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 2),
                nn.Softmax(dim=-1),
            )
        else:
            # Normalized fixed weights
            total = weight_resnet + weight_densenet
            self.register_buffer("w_resnet", torch.tensor(weight_resnet / total))
            self.register_buffer("w_densenet", torch.tensor(weight_densenet / total))

    def forward(
        self,
        probs_resnet: torch.Tensor,
        probs_densenet: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Fuses predictions from ResNet and DenseNet.
        
        Args:
            probs_resnet: (B, 5) probability distribution from ResNet model
            probs_densenet: (B, 5) probability distribution from DenseNet model
            
        Returns:
            fused_probs: (B, 5) combined probability distribution
            ensemble_info: Dict containing weights and individual stream predictions
        """
        if self.use_learnable_gate:
            cat_probs = torch.cat([probs_resnet, probs_densenet], dim=-1)
            gate_weights = self.gate(cat_probs)  # (B, 2)
            w_r = gate_weights[:, 0:1]
            w_d = gate_weights[:, 1:2]
            fused_probs = w_r * probs_resnet + w_d * probs_densenet
        else:
            w_r = self.w_resnet
            w_d = self.w_densenet
            fused_probs = w_r * probs_resnet + w_d * probs_densenet

        # Ensure probabilities strictly sum to 1
        fused_probs = fused_probs / fused_probs.sum(dim=-1, keepdim=True)

        best_stage_idx = int(torch.argmax(fused_probs[0]).item())
        best_stage_name = self.STAGE_NAMES[best_stage_idx]
        confidence = float(fused_probs[0, best_stage_idx].item())

        ensemble_info = {
            "predicted_stage": best_stage_name,
            "confidence": confidence,
            "fused_distribution": {
                name: float(fused_probs[0, i].item())
                for i, name in enumerate(self.STAGE_NAMES)
            },
            "weight_resnet": float(w_r.mean().item()) if isinstance(w_r, torch.Tensor) else float(w_r),
            "weight_densenet": float(w_d.mean().item()) if isinstance(w_d, torch.Tensor) else float(w_d),
        }

        return fused_probs, ensemble_info
