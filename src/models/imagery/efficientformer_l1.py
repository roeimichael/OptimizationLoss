# EfficientFormer-L1 wrapper for image classification.
# Hybrid conv+transformer backbone via timm.
# Input: (B, 3, 224, 224) -> logits (B, n_classes).

import torch
import torch.nn as nn
import timm


class EfficientFormerL1Classifier(nn.Module):

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientformer_l1',
            pretrained=pretrained,
            num_classes=n_classes,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
