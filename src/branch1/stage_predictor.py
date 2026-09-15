"""
Patient pN-Stage Classifier Module for Branch 1.
Maps the 25-D aggregated patient lymph-node profile vector to 5-class pN stage probabilities:
[pN0, pN0(i+), pN1mi, pN1, pN2].
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatientStagePredictor(nn.Module):
    """
    MLP classifier predicting patient-level pathological nodal (pN) stage.
    """

    STAGE_NAMES: List[str] = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]

    def __init__(
        self,
        in_dim: int = 25,
        hidden_dim: int = 64,
        num_stages: int = 5,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, num_stages),
        )

    def forward(self, patient_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patient_vector: Tensor of shape (B, 25) or (25,)
            
        Returns:
            stage_probs: (B, 5) Softmax probability distribution over 5 pN stages
            stage_logits: (B, 5) Unnormalized logits
        """
        if patient_vector.ndim == 1:
            patient_vector = patient_vector.unsqueeze(0)

        stage_logits = self.net(patient_vector)  # (B, 5)
        stage_probs = F.softmax(stage_logits, dim=-1)

        return stage_probs, stage_logits

    @classmethod
    def get_predicted_stage_name(cls, stage_probs: torch.Tensor) -> Tuple[str, float]:
        """
        Returns the stage name and confidence score for the argmax prediction.
        """
        if stage_probs.ndim == 2:
            stage_probs = stage_probs.squeeze(0)
        best_idx = int(torch.argmax(stage_probs).item())
        confidence = float(stage_probs[best_idx].item())
        return cls.STAGE_NAMES[best_idx], confidence
