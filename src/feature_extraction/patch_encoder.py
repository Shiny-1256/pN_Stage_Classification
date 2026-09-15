"""
Patch Feature Encoder.
Extracts deep representation embeddings for patches using pretrained models (ResNet50 / CTransPath).
Uses half-precision (fp16) to fit comfortably within RTX 4050 6GB VRAM.
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class PatchEncoder:
    """
    Feature extractor for histopathology patches.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        device: Optional[str] = None,
        use_amp: bool = True,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_amp = use_amp and (self.device.type == "cuda")
        self.model_name = model_name

        # Standard ImageNet / Pathology transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load backbone
        if model_name.lower() == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove classification head to output 2048-dim feature vectors
            self.model = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
            self.feature_dim = 2048
        elif model_name.lower() == "densenet121":
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.model = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def extract_features(self, patches: List[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """
        Extracts feature embeddings for a list of RGB patches.
        
        Args:
            patches: List of (H, W, 3) uint8 arrays.
            batch_size: Inference batch size.
            
        Returns:
            features: np.ndarray of shape (N, feature_dim), float32.
        """
        if len(patches) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        embeddings = []
        num_patches = len(patches)

        for i in range(0, num_patches, batch_size):
            batch_imgs = patches[i : i + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch_imgs]).to(self.device)

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                    feats = self.model(tensors)
            else:
                feats = self.model(tensors)

            embeddings.append(feats.cpu().float().numpy())

        return np.concatenate(embeddings, axis=0)
