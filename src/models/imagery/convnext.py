# ConvNeXt-Tiny wrapper for image classification.
# Uses torchvision pretrained weights with a custom dropout + linear head.
# Input: (B, 3, H, W) -> logits (B, n_classes).

import torch
import torch.nn as nn
from torchvision import models


class ConvNeXtTinyClassifier(nn.Module):

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        feat_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
