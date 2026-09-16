"""
Stain Normalization using the Macenko method
Standardizes H&E slide appearances across multiple hospital scanning centers
"""

from typing import Optional
import numpy as np


class MacenkoNormalizer:
    """
    Normalizes H&E stained pathology images using optical density (OD) decomposition
    and singular value decomposition (SVD) following the Macenko technique
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.15,
        target_io: float = 240.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.target_io = target_io

        # Standard reference H&E stain vectors (Macenko default values)
        self.he_ref = np.array([
            [0.5626, 0.2159], # Hematoxylin reference vector
            [0.7201, 0.8012], # Eosin reference vector 
            [0.4062, 0.5581], # Third channel reference vector (not used in H&E)
        ])
        # Standard reference maximum concentrations (99th percentile)
        self.max_c_ref = np.array([1.9705, 1.0308])

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalizes an RGB patch image
        Args: RGB image patch (H, W, 3) uint8.
        Returns: normalized_img: RGB normalized patch (H, W, 3) uint8.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        # If patch is mostly blank, skip expensive SVD
        if img.mean() > 240 or img.mean() < 15:
            return img

        img = img.astype(np.float64)
        h, w, c = img.shape

        # Step 1: Convert RGB to Optical Density (OD)
        # OD = -log((I + 1) / Io)
        img_reshaped = img.reshape((-1, 3))
        od = -np.log((img_reshaped + 1.0) / self.target_io)
        od = np.maximum(od, 0)

        # Step 2: Remove transparent pixels below optical density threshold beta
        od_hat = od[np.all(od > self.beta, axis=1)]
        if od_hat.shape[0] < 10:
            return img.astype(np.uint8)

        # Step 3: Compute SVD on the OD plane
        try:
            _, _, v = np.linalg.svd(od_hat, full_matrices=False)
            that = od_hat @ v[:2].T  # Project onto top 2 singular vectors

            # Step 4: Find extreme angles (alpha and 100-alpha percentiles)
            phi = np.arctan2(that[:, 1], that[:, 0])
            min_phi = np.percentile(phi, self.alpha)
            max_phi = np.percentile(phi, 100 - self.alpha)

            v1 = v[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
            v2 = v[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])

            # Order vectors so that v1 is Hematoxylin and v2 is Eosin
            if v1[0] > v2[0]:
                he_vectors = np.array([v1, v2]).T
            else:
                he_vectors = np.array([v2, v1]).T

            # Normalize columns
            he_vectors = he_vectors / np.linalg.norm(he_vectors, axis=0, keepdims=True)

            # Step 5: Extract stain concentrations (C = OD / HE)
            concentrations = np.linalg.lstsq(he_vectors, od.T, rcond=None)[0]

            # 99th percentile concentrations
            max_c = np.percentile(concentrations, 99, axis=1)
            max_c = np.maximum(max_c, 1e-4)

            # Normalize concentrations to target reference
            concentrations_norm = concentrations * (self.max_c_ref[:, None] / max_c[:, None])

            # Step 6: Reconstruct normalized image
            od_norm = self.he_ref @ concentrations_norm
            img_norm = self.target_io * np.exp(-od_norm)
            img_norm = np.clip(img_norm.T, 0, 255).reshape((h, w, 3))
            return img_norm.astype(np.uint8)

        except Exception:
            # In case of numerical instability, return original patch
            return img.astype(np.uint8)
