# Vision Transformer wrappers (timm). Includes ViT-Tiny and ViT-Small-32.
# Supports optional linear probe (freeze the backbone, train classifier only).

import torch
import torch.nn as nn

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


def _wrap(name: str, n_classes: int, pretrained: bool, dropout: float,
          linear_probe: bool):
    if timm is None:
        raise ImportError("timm is required for ViT backbones")
    model = timm.create_model(name, pretrained=pretrained, num_classes=n_classes,
                              drop_rate=dropout)
    if linear_probe:
        for p in model.parameters():
            p.requires_grad = False
        # timm ViTs expose `head` as the final classifier; unfreeze it.
        for p in model.head.parameters():
            p.requires_grad = True
    return model


class ViTTinyClassifier(nn.Module):
    """timm vit_tiny_patch16_224. ~5.7M params. Optional linear probe."""
    def __init__(self, n_classes: int = 7, pretrained: bool = False,
                 dropout: float = 0.3, linear_probe: bool = False, **kwargs):
        super().__init__()
        self.backbone = _wrap("vit_tiny_patch16_224", n_classes, pretrained,
                              dropout, linear_probe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ViTSmall32Classifier(nn.Module):
    """timm vit_small_patch32_224. ~22M params, larger patches => coarser
    tokens, lower compute, slower convergence than 16-patch variants."""
    def __init__(self, n_classes: int = 7, pretrained: bool = False,
                 dropout: float = 0.3, linear_probe: bool = False, **kwargs):
        super().__init__()
        self.backbone = _wrap("vit_small_patch32_224", n_classes, pretrained,
                              dropout, linear_probe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
