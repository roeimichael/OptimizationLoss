"""SmallCNN: ~100k params, 4-conv stack. A DIAGNOSTIC backbone, not a claim.

Restored 2026-09-15, having been removed on 2026-08-18 for the right reason --
no .tex file mentions it, so no paper result may rest on it, and FRAMEWORK
section 1 keeps the claimed set to four backbones. It is back for a narrower
job, and that job is elimination rather than evidence.

Every pretrained backbone we have memorises fmow2 in 3-5 epochs, which leaves
the constraint pushing a boundary cross-entropy has already stopped moving. We
cannot tell from those runs whether the constraint damages the boundary
because the mechanism is wrong or merely because it always arrives after the
boundary is set. A model small enough to still be learning at epoch 30
separates those two, and nothing else we have does.

So: results from this backbone are a CONTROL on the regime, and may be used to
rule a cause in or out. They may not become a headline, and a win here is not
a win -- FRAMEWORK section 1 still decides what the paper claims.
"""

import torch
import torch.nn as nn


class SmallCNNClassifier(nn.Module):
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
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
