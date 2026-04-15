# MobileNetV3-Large wrapper for image classification.
# Replaces the entire classifier to avoid double dropout from the original head.
# Input: (B, 3, H, W) -> logits (B, n_classes).

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights


class MobileNetV3Classifier(nn.Module):

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(1280, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
