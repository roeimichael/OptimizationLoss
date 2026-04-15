# ResNet18 wrapper for image classification.
# Uses torchvision pretrained weights with a custom dropout + linear head.
# Input: (B, 3, H, W) -> logits (B, n_classes).

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Classifier(nn.Module):

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
