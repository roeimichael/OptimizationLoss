"""Model registry supporting both tabular and imagery architectures."""

from typing import Any
import torch.nn as nn

from .tabular import BasicNN, TabularResNet, FTTransformer
from .imagery import ResNet18Classifier, ResNet50Classifier

# Registry: model_name -> (model_type, model_class)
MODEL_REGISTRY = {
    # Tabular models (input: flat feature vectors)
    'BasicNN': ('tabular', BasicNN),
    'TabularResNet': ('tabular', TabularResNet),
    'FTTransformer': ('tabular', FTTransformer),

    # Imagery models (input: (B, 3, H, W) image tensors)
    'ResNet18': ('imagery', ResNet18Classifier),
    'ResNet50': ('imagery', ResNet50Classifier),
}


def get_model(model_name: str, n_classes: int = 7, **kwargs: Any) -> nn.Module:
    """Instantiate a model by name.

    For tabular models, `input_dim` must be provided in kwargs.
    For imagery models, `input_dim` is ignored.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_type, model_class = MODEL_REGISTRY[model_name]

    if model_type == 'tabular':
        input_dim = kwargs.pop('input_dim')
        return model_class(input_dim=input_dim, n_classes=n_classes, **kwargs)
    else:
        # Imagery models don't use input_dim or hidden_dims
        kwargs.pop('input_dim', None)
        kwargs.pop('hidden_dims', None)
        return model_class(n_classes=n_classes, **kwargs)


def is_imagery_model(model_name: str) -> bool:
    """Check if a model is an imagery (CNN) model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name][0] == 'imagery'
