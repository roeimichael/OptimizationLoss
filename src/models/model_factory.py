"""Model factory for creating neural network architectures."""

from typing import Dict, Type, Any
import torch.nn as nn

from .tabular_resnet import TabularResNet
from .ft_transformer import FTTransformer
from .basic_nn import BasicNN
from src.utils.error_handler import logger


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'BasicNN': BasicNN,
    'TabularResNet': TabularResNet,
    'FTTransformer': FTTransformer
}


@logger()
def get_model(model_name: str, input_dim: int, n_classes: int = 3, **kwargs: Any) -> nn.Module:
    """
    Create and return a model instance by name.

    Args:
        model_name: Name of the model (must be in MODEL_REGISTRY)
        input_dim: Number of input features
        n_classes: Number of output classes
        **kwargs: Additional model-specific parameters

    Returns:
        Instantiated PyTorch model
    """
    model_class = MODEL_REGISTRY[model_name]
    return model_class(input_dim=input_dim, n_classes=n_classes, **kwargs)
