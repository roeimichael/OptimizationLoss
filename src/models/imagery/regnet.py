# RegNetY-400MF wrapper for image classification.
# Small RegNet (~4M params, group-conv+SE) -- non-MobileNet corroboration backbone.
# Confirmed: aider win-both, derm Hounie-only/Fior-tie (2026-05-28 Blackwell 4-seed paired).
# Replaces the final fc with a dropout + linear head.
# Input: (B, 3, H, W) -> logits (B, n_classes).

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import RegNet_Y_400MF_Weights


class RegNetY400MFClassifier(nn.Module):

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
        self.backbone = models.regnet_y_400mf(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
