"""
Model Factory for creating network architectures
Provides unified interface to all model types
"""
from .basic_nn import BasicNN
from .resnet56 import ResNet56
from .densenet121 import DenseNet121
from .inception_v3 import InceptionV3
from .vgg19 import VGG19


MODEL_REGISTRY = {
    'BasicNN': BasicNN,
    'ResNet56': ResNet56,
    'DenseNet121': DenseNet121,
    'InceptionV3': InceptionV3,
    'VGG19': VGG19
}


def get_model(model_name, input_dim, n_classes=3, **kwargs):
    """
    Factory function to create model instances

    Args:
        model_name: Name of the model architecture
        input_dim: Number of input features
        n_classes: Number of output classes (default: 3)
        **kwargs: Additional model-specific parameters (hidden_dims, dropout, etc.)

    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(input_dim=input_dim, n_classes=n_classes, **kwargs)


def list_available_models():
    """Return list of all available model names"""
    return list(MODEL_REGISTRY.keys())
