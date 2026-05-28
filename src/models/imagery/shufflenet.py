# ShuffleNetV2 (x1.0) wrapper for image classification.
# Replaces the final fc with a dropout + linear head.
# Input: (B, 3, H, W) -> logits (B, n_classes).

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ShuffleNet_V2_X1_0_Weights


class ShuffleNetV2Classifier(nn.Module):

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        self.backbone = models.shufflenet_v2_x1_0(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
