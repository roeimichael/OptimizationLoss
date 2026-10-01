"""Pinned modern backbones have explicit transforms and fail closed on weights."""

import pytest
from PIL import Image

from tralo.tabular_backbones import (PINNED, image_transforms,
                                     verified_pretrained_weight)


def test_all_backbones_have_binary_input_transforms_and_fixed_weight_hashes(tmp_path):
    image = Image.new("RGB", (256, 256), (80, 100, 120))
    assert set(PINNED) == {"mobilenet_v3_large", "vit_b_16", "convnext_tiny"}
    for name in PINNED:
        training, evaluation = image_transforms(name)
        assert tuple(training(image).shape) == (3, 224, 224)
        assert tuple(evaluation(image).shape) == (3, 224, 224)
        with pytest.raises(RuntimeError, match="missing or linked"):
            verified_pretrained_weight(name, cache_root=tmp_path)
    with pytest.raises(ValueError, match="unfrozen"):
        image_transforms("efficientnet")
