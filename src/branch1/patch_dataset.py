"""
Patch Dataset for Branch 1 Deep Learning Models.
Loads preprocessed 224x224 patches, coordinates, and labels directly from HDF5 (.h5) files.
Supports data augmentation, class balancing, and memory-mapped random access.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Callable
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class H5PatchDataset(Dataset):
    """
    Dataset that indexes and loads patches across multiple HDF5 slide archives.
    """

    def __init__(
        self,
        h5_dir: str | Path,
        slide_ids: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        is_training: bool = True,
    ):
        self.h5_dir = Path(h5_dir)
        if slide_ids is not None:
            self.h5_files = [self.h5_dir / f"{sid}.h5" for sid in slide_ids if (self.h5_dir / f"{sid}.h5").exists()]
        else:
            self.h5_files = sorted(list(self.h5_dir.glob("*.h5")))

        # Standard pathology normalization transforms
        if transform is not None:
            self.transform = transform
        else:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        # Build global patch index across all H5 files
        self.patch_index: List[Tuple[int, int]] = []  # List of (file_idx, patch_idx_within_file)
        self.labels: List[int] = []

        for f_idx, h5_p in enumerate(self.h5_files):
            with h5py.File(h5_p, "r") as f:
                n_patches = f["images"].shape[0]
                lbls = f["labels"][:]
                for p_idx in range(n_patches):
                    self.patch_index.append((f_idx, p_idx))
                    self.labels.append(int(lbls[p_idx]))

        self.labels = np.array(self.labels, dtype=np.int64)

        # Cache open file handles to avoid repeated disk open overhead
        self._file_handles = {}

    def _get_file_handle(self, file_idx: int) -> h5py.File:
        if file_idx not in self._file_handles:
            self._file_handles[file_idx] = h5py.File(self.h5_files[file_idx], "r")
        return self._file_handles[file_idx]

    def __len__(self) -> int:
        return len(self.patch_index)

    def __getitem__(self, idx: int) -> dict:
        f_idx, p_idx = self.patch_index[idx]
        h5_f = self._get_file_handle(f_idx)

        image_np = h5_f["images"][p_idx]  # (224, 224, 3) uint8
        coord = h5_f["coords"][p_idx]      # (2,) int32
        label = int(h5_f["labels"][p_idx])

        # Apply transforms -> (3, 224, 224)
        tensor = self.transform(image_np)

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "coord": torch.tensor(coord, dtype=torch.long),
            "slide_id": self.h5_files[f_idx].stem,
        }

    def close(self):
        for f in self._file_handles.values():
            f.close()
        self._file_handles.clear()

    def __del__(self):
        self.close()
