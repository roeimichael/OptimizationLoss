# Model registry for imagery architectures.
# Dispatches model instantiation by name via MODEL_REGISTRY.

from typing import Any
import torch.nn as nn

from .imagery import (
    ResNet18Classifier, MobileNetV3Classifier,
    EfficientNetB0Classifier, ConvNeXtTinyClassifier,
)

MODEL_REGISTRY = {
    'ResNet18': ResNet18Classifier,
    'MobileNetV3': MobileNetV3Classifier,
    'EfficientNetB0': EfficientNetB0Classifier,
    'ConvNeXtTiny': ConvNeXtTinyClassifier,
}


def get_model(model_name: str, n_classes: int = 7, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name]
    kwargs.pop('input_dim', None)
    kwargs.pop('hidden_dims', None)
    return model_class(n_classes=n_classes, **kwargs)
