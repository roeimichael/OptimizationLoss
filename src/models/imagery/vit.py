# ViT-B/16 wrapper for image classification (SOTA transformer backbone, ~86M params).
# Replaces the classification head; expects 224x224 ImageNet-normalized input.
# Input: (B, 3, 224, 224) -> logits (B, n_classes).
# NOTE: re-introduced 2026-06-10 to test transformer-class corroboration on the new
# (harder) OctMNIST dataset + the deployment pillar. See docs/REJECTED.md: ViT-S/B
# previously saturated tissue/derm/aider (train-acc -> 1.0 in 1-2 epochs), masking the
# cc-F1 advantage. Probe the warmup band before any full sweep.

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights


class ViTB16Classifier(nn.Module):

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        hidden = self.backbone.heads.head.in_features  # 768 for ViT-B/16
        self.backbone.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
