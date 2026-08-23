"""Model registry for imagery architectures.

The four backbones the paper claims, and nothing else. ShuffleNetV2, TinyCNN,
SmallCNN and MediumCNN were removed on 2026-08-18: none appears in any .tex
file, so no result rests on them (see docs/FRAMEWORK.md section 1).
"""
from typing import Any

import torch.nn as nn

from .imagery import (
    MobileNetV3Classifier,
    MobileNetV2Classifier,
    RegNetY400MFClassifier,
    ViTB16Classifier,
)

MODEL_REGISTRY = {
    'MobileNetV3': MobileNetV3Classifier,      # headline
    'MobileNetV2': MobileNetV2Classifier,
    'RegNetY400MF': RegNetY400MFClassifier,
    'ViTB16': ViTB16Classifier,                # the non-CNN check
}


def get_model(model_name: str, n_classes: int, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY)}")
    kwargs.pop('input_dim', None)
    return MODEL_REGISTRY[model_name](n_classes=n_classes, **kwargs)
