"""Model factory: create model instances by name from registry."""

from typing import Dict, Type, Any
import torch.nn as nn

from .tabular_resnet import TabularResNet
from .ft_transformer import FTTransformer
from .basic_nn import BasicNN

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'BasicNN': BasicNN,
    'TabularResNet': TabularResNet,
    'FTTransformer': FTTransformer
}


def get_model(model_name: str, input_dim: int, n_classes: int = 3, **kwargs: Any) -> nn.Module:
    """Create and return a model instance by name."""
    return MODEL_REGISTRY[model_name](input_dim=input_dim, n_classes=n_classes, **kwargs)
