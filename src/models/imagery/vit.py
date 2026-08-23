# ViT-B/16 wrapper for image classification (SOTA transformer backbone, ~86M params).
# Replaces the classification head; expects 224x224 ImageNet-normalized input.
# Input: (B, 3, 224, 224) -> logits (B, n_classes).
# NOTE: one of the four backbones the paper claims. It was re-introduced 2026-06-10
# for transformer-class corroboration, and the reason it had been dropped is the
# reason to watch it now: ViT-S/B SATURATED the old benchmarks, train-acc -> 1.0 in
# 1-2 epochs, and every method then landed within +-0.005 F1 because the constraint
# phase had no slack to redistribute. Recorded in
# docs/archive/REJECTED_full_2026-08-18.md ("ViT-S and ConvNeXt-T").
#
# THIS IS LIVE, NOT HISTORY. iwildcam's warm-up reaches 95.6% train accuracy in ONE
# epoch, so the saturation this note describes is exactly what FRAMEWORK 2(p) step 0b
# makes mandatory before any iwc1 contrast is read. Probe the warm-up band -- with
# `scripts.reachability`, on the TEST predictions -- before any full sweep.

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
