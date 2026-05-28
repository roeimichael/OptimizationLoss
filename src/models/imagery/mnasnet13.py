# MNASNet1.3 wrapper for image classification.
# Larger MNASNet variant (1.3 width mult). Replaces entire classifier to avoid
# double dropout from the original head (per MobileNetV3 fix).
# Input: (B, 3, H, W) -> logits (B, n_classes).

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MNASNet1_3_Weights


class MNASNet13Classifier(nn.Module):

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = MNASNet1_3_Weights.DEFAULT if pretrained else None
        self.backbone = models.mnasnet1_3(weights=weights)
        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
