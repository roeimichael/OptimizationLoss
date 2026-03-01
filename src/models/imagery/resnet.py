"""ResNet wrappers for image classification with torchvision pretrained weights."""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Classifier(nn.Module):
    """ResNet18 with custom classification head."""

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        feat_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, 3, H, W) -> logits (B, n_classes)."""
        return self.backbone(x)
