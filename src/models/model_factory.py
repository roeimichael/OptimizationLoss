# Model registry for imagery architectures.
# Dispatches model instantiation by name via MODEL_REGISTRY.

from typing import Any
import torch.nn as nn

from .imagery import (
    MobileNetV3Classifier,
    MobileNetV2Classifier,
    ShuffleNetV2Classifier,
    RegNetY400MFClassifier,
    TinyCNNClassifier,
    SmallCNNClassifier,
    MediumCNNClassifier,
)

MODEL_REGISTRY = {
    'MobileNetV3': MobileNetV3Classifier,
    'MobileNetV2': MobileNetV2Classifier,
    'ShuffleNetV2': ShuffleNetV2Classifier,
    'RegNetY400MF': RegNetY400MFClassifier,
    'TinyCNN': TinyCNNClassifier,
    'SmallCNN': SmallCNNClassifier,
    'MediumCNN': MediumCNNClassifier,
}


def get_model(model_name: str, n_classes: int, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name]
    kwargs.pop('input_dim', None)
    return model_class(n_classes=n_classes, **kwargs)
