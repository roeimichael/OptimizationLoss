# MobileNetV3-Large wrapper for image classification.
# Input: (B, 3, H, W) -> logits (B, n_classes).
#
# Keeps the pretrained 960->1280 projection and replaces only the final layer,
# which is what MobileNetV2, RegNetY400MF and ViTB16 all do. Rebuilding the
# whole classifier -- the previous behaviour, done to avoid the original head's
# double dropout -- threw away that projection and started it from random.
#
# That mattered more than it looks. The projection is only trained during
# warm-up, and the protocol gives trained arms ONE warm-up epoch against the
# post-hoc arms' thirty. So on the headline backbone the trained arms began
# from a materially worse model than the baseline they are measured against --
# a bias in the direction of the headline comparison, on the headline backbone.
# The double dropout is avoided by setting the EXISTING Dropout's p instead.

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights


class MobileNetV3Classifier(nn.Module):

    def __init__(self, n_classes: int = 7, pretrained: bool = False, dropout: float = 0.3, **kwargs):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)
        head = self.backbone.classifier
        for layer in head:
            if isinstance(layer, nn.Dropout):
                layer.p = dropout          # reuse it; do not add a second one
        head[-1] = nn.Linear(head[-1].in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
