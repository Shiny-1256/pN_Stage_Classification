"""
Tissue Mask Generation for WSI Preprocessing
Performs background removal, Otsu thresholding, and morphological filtering
"""

from typing import Tuple
import cv2
import numpy as np

class TissueDetector:
    """
    Detects valid tissue regions on downsampled whole slide thumbnails
    eliminates glass background, pen marks, and air bubbles.
    """
    def __init__(
        self,
        saturation_threshold: int = 15, # color saturation - 0 to 255
        min_tissue_area_px: int = 1000, # min area of island to be considered tissue
    ):
        self.saturation_threshold = saturation_threshold
        self.min_tissue_area_px = min_tissue_area_px

    def segment_tissue(self, thumbnail: np.ndarray) -> np.ndarray:
        """
        Segments tissue from thumbnail image
        Args: RGB thumbnail numpy array (H, W, 3), dtype=uint8.
        Returns: binary_mask - 2D uint8 array (H, W) where 1 indicates tissue and 0 indicates background.
        """
        if thumbnail.ndim != 3 or thumbnail.shape[2] != 3:
            raise ValueError(f"Thumbnail must be RGB (H, W, 3), got shape {thumbnail.shape}")

        # RGB to HSV color space
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # In H&E tissue, saturation is higher than blank glass (which has saturation near 0)
        # Background glass is bright with near-zero saturation
        # Automatic Otsu thresholding on saturation channel
        _, sat_thresh = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Baseline saturation cutoff (guards against low contrast slides)
        manual_sat = (sat > self.saturation_threshold).astype(np.uint8) * 255

        # Also filter out very dark artifacts (e.g., sharp pen marks or slide boundaries)
        not_pitch_black = (val > 20).astype(np.uint8) * 255

        combined_mask = cv2.bitwise_and(cv2.bitwise_or(sat_thresh, manual_sat), not_pitch_black)

        # Morphological operations to close small internal holes and remove isolated pixel noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        # Filter out tiny connected components (dust/specs)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        filtered_mask = np.zeros_like(cleaned)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_tissue_area_px:
                filtered_mask[labels == i] = 1

        return filtered_mask.astype(np.uint8)
