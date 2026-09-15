"""
Spatial 3x3 Neighborhood Builder for WSI Patches.
Given the Level 0 (x, y) coordinates of all valid patches in a slide,
constructs an efficient 8-neighbor adjacency lookup table for every patch.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import torch


class SpatialNeighborhoodBuilder:
    """
    Builds the 8-neighbor spatial adjacency graph for patches in a WSI.
    """

    # 8 neighbor offsets relative to center (dx, dy):
    # Top-Left, Top, Top-Right, Left, Right, Bottom-Left, Bottom, Bottom-Right
    NEIGHBOR_OFFSETS = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]

    def __init__(self, patch_size: int = 256):
        self.patch_size = patch_size

    def build_neighborhood_indices(
        self,
        coords: np.ndarray | torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs adjacency indices for N patches.
        
        Args:
            coords: Array or Tensor of shape (N, 2) where each row is (x, y) at Level 0.
            
        Returns:
            neighbor_indices: Int64 Tensor of shape (N, 8) with indices in [0, N-1].
                              Missing neighbors default to the central patch index.
            neighbor_mask: Bool Tensor of shape (N, 8) where True indicates a valid neighbor,
                           and False indicates a missing/boundary neighbor.
        """
        if isinstance(coords, torch.Tensor):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = np.asarray(coords)

        num_patches = len(coords_np)
        # Fast hash map from (x, y) -> patch_index
        coord_map: Dict[Tuple[int, int], int] = {
            (int(coords_np[i, 0]), int(coords_np[i, 1])): i
            for i in range(num_patches)
        }

        neighbor_indices = np.zeros((num_patches, 8), dtype=np.int64)
        neighbor_mask = np.zeros((num_patches, 8), dtype=bool)

        for i in range(num_patches):
            cx = int(coords_np[i, 0])
            cy = int(coords_np[i, 1])

            for k, (dx, dy) in enumerate(self.NEIGHBOR_OFFSETS):
                nx = cx + dx * self.patch_size
                ny = cy + dy * self.patch_size

                if (nx, ny) in coord_map:
                    neighbor_indices[i, k] = coord_map[(nx, ny)]
                    neighbor_mask[i, k] = True
                else:
                    # Missing neighbor: point to center patch, but mask is False
                    neighbor_indices[i, k] = i
                    neighbor_mask[i, k] = False

        return torch.from_numpy(neighbor_indices), torch.from_numpy(neighbor_mask)
