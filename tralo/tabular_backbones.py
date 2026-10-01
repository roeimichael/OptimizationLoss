"""Pinned image-only backbones and preprocessing for metadata quotas."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms


PINNED = {
    "mobilenet_v3_large": (
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "mobilenet_v3_large-5c1a4163.pth",
        "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"),
    "vit_b_16": (
        models.ViT_B_16_Weights.IMAGENET1K_V1,
        "vit_b_16-c867db91.pth",
        "c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9"),
    "convnext_tiny": (
        models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_tiny-983f1562.pth",
        "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"),
}


def configure_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)


def verified_pretrained_weight(backbone, *, cache_root=None):
    if backbone not in PINNED:
        raise ValueError("unfrozen image backbone")
    enum, filename, expected = PINNED[backbone]
    path = ((Path(cache_root) if cache_root is not None else
             Path(torch.hub.get_dir()) / "checkpoints") / filename)
    h = hashlib.sha256()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("pinned image weight missing or linked")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != expected:
        raise RuntimeError("pinned image weight bytes changed")
    return {"file": str(path), "sha256": expected,
            "weight_enum": str(enum), "url": enum.url}


def make_binary_model(backbone):
    """Load a hash-verified ImageNet backbone and replace only its classifier."""
    weight = verified_pretrained_weight(backbone)
    enum = PINNED[backbone][0]
    if backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=enum)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif backbone == "vit_b_16":
        model = models.vit_b_16(weights=enum)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    else:
        model = models.convnext_tiny(weights=enum)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    return model, weight


def image_transforms(backbone):
    if backbone not in PINNED:
        raise ValueError("unfrozen image backbone")
    evaluation = PINNED[backbone][0].transforms()
    training = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return training, evaluation
