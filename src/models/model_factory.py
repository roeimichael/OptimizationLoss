from typing import Dict, Type, Any
import torch.nn as nn

# Tabular-specific models
from .tabular_resnet import TabularResNet
from .ft_transformer import FTTransformer

# Legacy/baseline models
from .basic_nn import BasicNN
from .resnet56 import ResNet56
from .densenet121 import DenseNet121
from .inception_v3 import InceptionV3
from .vgg19 import VGG19

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    # Tabular-specific models (current experiments)
    'BasicNN': BasicNN,  # Simple MLP baseline for tabular data
    'TabularResNet': TabularResNet,
    'FTTransformer': FTTransformer,

    # Legacy models (vision-based, kept for backward compatibility)
    'ResNet56': ResNet56,
    'DenseNet121': DenseNet121,
    'InceptionV3': InceptionV3,
    'VGG19': VGG19
}

def get_model(model_name: str, input_dim: int, n_classes: int = 3, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name]
    return model_class(input_dim=input_dim, n_classes=n_classes, **kwargs)
