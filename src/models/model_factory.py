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
    SmallCNNClassifier,
    ViTB16Classifier,
)

MODEL_REGISTRY = {
    # THE HEADLINE IS ViTB16, fixed a priori 2026-08-20 (FRAMEWORK 1-pre) so
    # that a win found on another backbone cannot be promoted after the fact.
    # These comments said the opposite until 2026-09-01.
    'MobileNetV3': MobileNetV3Classifier,
    'MobileNetV2': MobileNetV2Classifier,
    'RegNetY400MF': RegNetY400MFClassifier,
    'ViTB16': ViTB16Classifier,                # THE HEADLINE
    # DIAGNOSTIC ONLY, restored 2026-09-15. Small enough to still be learning
    # after the pretrained backbones have memorised, which is the only way to
    # separate "the constraint damages the boundary" from "the constraint always
    # arrives after the boundary has stopped moving". Never a paper claim: see
    # src/models/imagery/smallcnn.py and FRAMEWORK section 1.
    'SmallCNN': SmallCNNClassifier,
}


def get_model(model_name: str, n_classes: int, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](n_classes=n_classes, **kwargs)
