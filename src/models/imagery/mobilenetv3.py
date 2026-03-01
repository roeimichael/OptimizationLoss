"""MobileNetV3-Large wrapper for image classification with torchvision pretrained weights."""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights


class MobileNetV3Classifier(nn.Module):
    """MobileNetV3-Large with custom classification head (~5.4M params)."""

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)
        feat_dim = self.backbone.classifier[-1].in_features  # 1280
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, 3, H, W) -> logits (B, n_classes)."""
        return self.backbone(x)
