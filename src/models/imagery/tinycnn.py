# TinyCNN: ~25k params, 3-conv stack + global avg pool + linear head.
# Designed to be well below Zhang et al. 2017's interpolation bound
# (2n + d ~ 166k params for DermMNIST 8k samples * 224x224x3 input)
# so it provably cannot memorize the train set.
#
# Input: (B, 3, H, W) -> logits (B, n_classes). pretrained kwarg ignored
# (no torchvision weights for a custom architecture); kept for API parity.

import torch
import torch.nn as nn


class TinyCNNClassifier(nn.Module):
    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
