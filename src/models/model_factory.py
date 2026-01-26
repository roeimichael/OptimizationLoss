from typing import Dict, Type, Any
import torch.nn as nn
from .tabular_resnet import TabularResNet
from .ft_transformer import FTTransformer
from .basic_nn import BasicNN


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    # Tabular-specific models (current experiments)
    'BasicNN': BasicNN,  # Simple MLP baseline for tabular data
    'TabularResNet': TabularResNet,
    'FTTransformer': FTTransformer
}

def get_model(model_name: str, input_dim: int, n_classes: int = 3, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name]
    return model_class(input_dim=input_dim, n_classes=n_classes, **kwargs)
