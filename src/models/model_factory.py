# Model registry for imagery architectures.
# Dispatches model instantiation by name via MODEL_REGISTRY.

from typing import Any
import torch.nn as nn

from .imagery import (
    MobileNetV3Classifier, ConvNeXtTinyClassifier,
    MobileNetV2Classifier, ShuffleNetV2Classifier,
    RegNetY400MFClassifier, MobileViTSClassifier,
    EfficientFormerL1Classifier, MNASNet13Classifier,
    ConvNeXtPicoClassifier,
    ViTTinyClassifier, ViTSmall32Classifier,
)

MODEL_REGISTRY = {
    'MobileNetV3': MobileNetV3Classifier,
    'ConvNeXtTiny': ConvNeXtTinyClassifier,
    'MobileNetV2': MobileNetV2Classifier,
    'ShuffleNetV2': ShuffleNetV2Classifier,
    'RegNetY400MF': RegNetY400MFClassifier,
    'MobileViTS': MobileViTSClassifier,
    'EfficientFormerL1': EfficientFormerL1Classifier,
    'MNASNet13': MNASNet13Classifier,
    'ConvNeXtPico': ConvNeXtPicoClassifier,
    'ViTTiny': ViTTinyClassifier,
    'ViTSmall32': ViTSmall32Classifier,
}


def get_model(model_name: str, n_classes: int, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name]
    # Strip vestigial tabular-era plumbing that callers still pass through.
    kwargs.pop('input_dim', None)
    return model_class(n_classes=n_classes, **kwargs)
