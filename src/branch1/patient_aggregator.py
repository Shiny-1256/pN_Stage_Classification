"""
Patient-Level Multi-Slide Aggregator for Branch 1.
Aggregates patch-level predictions across all examined lymph node slides (typically 5)
into a standardized, permutation-invariant patient profile vector.
"""

from typing import List, Dict, Any
import numpy as np
import torch


class PatientSlideAggregator:
    """
    Computes slide-level metastatic burden statistics and aggregates them across
    the 5 lymph nodes of a patient into a fixed-length feature representation.
    """

    def __init__(self, top_k_percentiles: List[float] = [0.01, 0.05]):
        self.top_k_percentiles = top_k_percentiles

    def extract_slide_features(self, patch_tumor_probs: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Computes 5 quantitative metastatic burden descriptors from a slide's patch probabilities.
        
        Args:
            patch_tumor_probs: Array/Tensor of shape (N,) containing P(tumor | patch) in [0, 1].
            
        Returns:
            slide_feat: (5,) array containing:
                [0]: Maximum tumor probability (focal peak)
                [1]: Top 1% mean tumor probability
                [2]: Top 5% mean tumor probability
                [3]: Fraction of positive patches (P >= 0.5, total burden)
                [4]: Global mean patch probability
        """
        if isinstance(patch_tumor_probs, torch.Tensor):
            probs = patch_tumor_probs.detach().cpu().numpy()
        else:
            probs = np.asarray(patch_tumor_probs)

        N = len(probs)
        if N == 0:
            return np.zeros(5, dtype=np.float32)

        sorted_probs = np.sort(probs)[::-1]  # Descending order

        # 1. Max probability
        max_p = float(sorted_probs[0])

        # 2. Top 1% mean
        k1 = max(1, int(np.ceil(0.01 * N)))
        top1_mean = float(np.mean(sorted_probs[:k1]))

        # 3. Top 5% mean
        k5 = max(1, int(np.ceil(0.05 * N)))
        top5_mean = float(np.mean(sorted_probs[:k5]))

        # 4. Positive patch fraction (tumor burden)
        pos_ratio = float(np.count_nonzero(probs >= 0.5)) / float(N)

        # 5. Global mean
        global_mean = float(np.mean(probs))

        return np.array([max_p, top1_mean, top5_mean, pos_ratio, global_mean], dtype=np.float32)

    def aggregate_patient_profile(
        self,
        slide_feature_list: List[np.ndarray],
        max_nodes: int = 5,
    ) -> np.ndarray:
        """
        Aggregates slide feature vectors across a patient's lymph nodes into a sorted,
        permutation-invariant patient feature vector.
        
        Args:
            slide_feature_list: List of (5,) feature arrays (one per node slide).
            max_nodes: Target number of lymph nodes (default 5 for CAMELYON17).
            
        Returns:
            patient_vector: Array of shape (5 * max_nodes,) = (25,) float32.
        """
        if len(slide_feature_list) == 0:
            return np.zeros(5 * max_nodes, dtype=np.float32)

        # Sort slides in descending order of tumor burden (index 3) and max probability (index 0)
        sorted_slides = sorted(
            slide_feature_list,
            key=lambda sf: (sf[3], sf[0]),
            reverse=True,
        )

        # Pad with zeros if patient has fewer than max_nodes
        padded = []
        for i in range(max_nodes):
            if i < len(sorted_slides):
                padded.append(sorted_slides[i])
            else:
                padded.append(np.zeros(5, dtype=np.float32))

        # Flatten into 25-D vector
        patient_vector = np.concatenate(padded, axis=0)
        return patient_vector
