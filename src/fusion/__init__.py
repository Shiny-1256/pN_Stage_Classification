"""
Late Fusion Module (SuperNet Meta-Learner)
Aggregates predictions from Branch 1, Branch 2, and Branch 3 into a consensus pN-stage.
"""

from .late_fusion_model import SuperNetConsensusClassifier

__all__ = ["SuperNetConsensusClassifier"]
