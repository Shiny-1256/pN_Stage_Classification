"""
WSI Pyramid Reader using tifffile + zarr.
Reads multi-resolution whole slide images without loading gigapixels into memory.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import tifffile
import zarr


class WSIReader:
    """
    Wrapper for reading multi-resolution pyramidal Whole Slide Images (WSI) in TIFF format.
    Supports random-access tile reading at Level 0 and downsampled thumbnail extraction.
    """

    def __init__(self, slide_path: str | Path):
        self.slide_path = Path(slide_path)
        if not self.slide_path.exists():
            raise FileNotFoundError(f"Slide not found: {self.slide_path}")

        self._tif = tifffile.TiffFile(str(self.slide_path))
        self.series = self._tif.series[0]
        self._store = self.series.aszarr()
        self._zarr_group = zarr.open(self._store, mode="r")

        # Level keys are strings: '0', '1', '2', ...
        self.num_levels = len(self.series.levels)
        self.level_shapes = [self._zarr_group[str(i)].shape for i in range(self.num_levels)]

        # Base dimensions at Level 0 (Height, Width)
        self.height, self.width = self.level_shapes[0][:2]

        # Calculate downsample factors relative to Level 0
        self.level_downsamples = [
            self.height / shape[0] for shape in self.level_shapes
        ]

    def get_thumbnail(self, target_level: int = 7) -> np.ndarray:
        """
        Loads a downsampled thumbnail of the entire slide for fast tissue segmentation.
        Uses target_level (or closest available level).
        """
        level = min(target_level, self.num_levels - 1)
        zarr_level = self._zarr_group[str(level)]
        thumbnail = np.array(zarr_level[:])
        return thumbnail

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Reads a patch from the WSI.
        
        Args:
            location: (x, y) coordinates at Level 0.
            level: Pyramidal level to read from (0 is highest magnification).
            size: (width, height) of the patch to read at specified level.
            
        Returns:
            np.ndarray: RGB image patch of shape (height, width, 3), dtype=uint8.
        """
        x_lvl0, y_lvl0 = location
        w, h = size

        downsample = self.level_downsamples[level]
        x_target = int(round(x_lvl0 / downsample))
        y_target = int(round(y_lvl0 / downsample))

        lvl_array = self._zarr_group[str(level)]
        max_h, max_w = lvl_array.shape[:2]

        # Slice bounds
        y1 = max(0, min(y_target, max_h))
        y2 = max(0, min(y_target + h, max_h))
        x1 = max(0, min(x_target, max_w))
        x2 = max(0, min(x_target + w, max_w))

        patch = np.array(lvl_array[y1:y2, x1:x2])

        # Pad if tile is near edge
        if patch.shape[0] != h or patch.shape[1] != w:
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            actual_h, actual_w = patch.shape[:2]
            if actual_h > 0 and actual_w > 0:
                padded[:actual_h, :actual_w] = patch
            return padded

        return patch

    def close(self):
        if self._tif is not None:
            self._tif.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
