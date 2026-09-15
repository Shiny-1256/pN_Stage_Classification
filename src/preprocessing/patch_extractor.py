"""
Patch Extractor for Whole Slide Images.
Computes non-overlapping patch grids over valid tissue, extracts patches,
and pairs each patch with spatial (x, y) coordinates and ground-truth tumor labels.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.utils.wsi_reader import WSIReader
from src.utils.annotation_parser import AnnotationParser
from src.preprocessing.tissue_mask import TissueDetector


class PatchExtractor:
    """
    Manages grid patch coordinate sampling and patch extraction from WSIs.
    """

    def __init__(
        self,
        patch_size: int = 224,
        min_tissue_ratio: float = 0.50,
        thumbnail_level: int = 7,
    ):
        self.patch_size = patch_size
        self.min_tissue_ratio = min_tissue_ratio
        self.thumbnail_level = thumbnail_level
        self.tissue_detector = TissueDetector()

    def generate_patch_coordinates(
        self,
        wsi_reader: WSIReader,
        annotation_parser: Optional[AnnotationParser] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Scans thumbnail tissue mask and computes Level 0 coordinates for patches that fall on tissue.
        
        Returns:
            mask: 2D binary tissue mask at thumbnail level.
            patch_records: List of dicts containing {x, y, label, overlap_ratio}.
        """
        # Step 1: Extract thumbnail and compute tissue mask
        thumb_level = min(self.thumbnail_level, wsi_reader.num_levels - 1)
        thumbnail = wsi_reader.get_thumbnail(target_level=thumb_level)
        mask = self.tissue_detector.segment_tissue(thumbnail)

        mask_h, mask_w = mask.shape
        downsample = wsi_reader.level_downsamples[thumb_level]

        # Patch size scaled to thumbnail dimensions
        patch_w_thumb = self.patch_size / downsample
        patch_h_thumb = self.patch_size / downsample

        patch_records = []

        # Step 2: Grid search across the slide
        y_max = wsi_reader.height - self.patch_size
        x_max = wsi_reader.width - self.patch_size

        for y0 in range(0, y_max, self.patch_size):
            # Corresponding coordinates on the mask
            my1 = int(round(y0 / downsample))
            my2 = int(round((y0 + self.patch_size) / downsample))
            if my1 >= mask_h:
                continue
            my2 = min(my2, mask_h)

            for x0 in range(0, x_max, self.patch_size):
                mx1 = int(round(x0 / downsample))
                mx2 = int(round((x0 + self.patch_size) / downsample))
                if mx1 >= mask_w:
                    continue
                mx2 = min(mx2, mask_w)

                # Check tissue coverage ratio in this patch
                mask_crop = mask[my1:my2, mx1:mx2]
                if mask_crop.size == 0:
                    continue

                tissue_ratio = np.count_nonzero(mask_crop) / mask_crop.size

                if tissue_ratio >= self.min_tissue_ratio:
                    # Determine tumor label if annotation parser is provided
                    label, overlap = 0, 0.0
                    if annotation_parser is not None and len(annotation_parser.polygons) > 0:
                        label, overlap = annotation_parser.is_tumor_patch(
                            x=x0, y=y0, patch_size=self.patch_size
                        )

                    patch_records.append({
                        "x": x0,
                        "y": y0,
                        "label": label,
                        "overlap_ratio": float(overlap),
                        "tissue_ratio": float(tissue_ratio),
                    })

        return mask, patch_records
