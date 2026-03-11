# Model registry supporting both tabular and imagery architectures.
# Dispatches model instantiation by name via MODEL_REGISTRY.
# Tabular models require input_dim; imagery models ignore it.

from typing import Any
import torch.nn as nn

from .tabular import BasicNN, TabularResNet, FTTransformer
from .imagery import ResNet18Classifier, MobileNetV3Classifier, EfficientNetB0Classifier, ConvNeXtTinyClassifier

MODEL_REGISTRY = {
    'BasicNN': ('tabular', BasicNN),
    'TabularResNet': ('tabular', TabularResNet),
    'FTTransformer': ('tabular', FTTransformer),
    'ResNet18': ('imagery', ResNet18Classifier),
    'MobileNetV3': ('imagery', MobileNetV3Classifier),
    'EfficientNetB0': ('imagery', EfficientNetB0Classifier),
    'ConvNeXtTiny': ('imagery', ConvNeXtTinyClassifier),
}


def get_model(model_name: str, n_classes: int = 7, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_type, model_class = MODEL_REGISTRY[model_name]

    if model_type == 'tabular':
        input_dim = kwargs.pop('input_dim')
        kwargs.pop('pretrained', None)
        return model_class(input_dim=input_dim, n_classes=n_classes, **kwargs)
    else:
        kwargs.pop('input_dim', None)
        kwargs.pop('hidden_dims', None)
        return model_class(n_classes=n_classes, **kwargs)


def is_imagery_model(model_name: str) -> bool:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name][0] == 'imagery'
