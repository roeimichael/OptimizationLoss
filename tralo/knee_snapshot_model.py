"""Offline, exact-byte ImageNet construction for the approved knee core.

This component does not issue campaign gates, claim a seed or allocate a GPU.
The caller must enforce those boundaries before constructing a campaign model.
"""

import hashlib
import io
from pathlib import Path

WEIGHT_SHA256 = '5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997'
WEIGHT_URL = 'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth'
WEIGHT_ENUM = 'MobileNet_V3_Large_Weights.IMAGENET1K_V2'


def make_model(weight_path, seed):
    """Authenticate the same bytes loaded, then preserve Yuval's RNG draw order.

    Torchvision first initializes the ordinary 1000-class architecture, loads
    ImageNet weights, and replaces only classifier[3] with a fresh five-way head.
    This follows that sequence without a URL request or a second weight read.
    CPU RNG is intentionally not restored: construction has the native behavior.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('model seed must be an explicit uint32 integer')
    path = Path(weight_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('pinned pretrained weight is missing or linked')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != WEIGHT_SHA256:
        raise ValueError('pinned pretrained weight bytes differ')
    import torch
    import torchvision
    from torchvision import models

    if torch.get_default_dtype() != torch.float32:
        raise ValueError('approved constructor requires the default float32 dtype')
    weights = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    torch.manual_seed(seed)
    before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    with torch.device('cpu'):
        model = models.mobilenet_v3_large(weights=None)
        model.load_state_dict(weights, strict=True)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 5)
    if (any(p.dtype != torch.float32 or p.device.type != 'cpu' or not p.requires_grad
            for p in model.parameters()) or model.classifier[3].out_features != 5):
        raise RuntimeError('approved unfrozen FP32 model construction differs')
    receipt = dict(backbone='mobilenet_v3_large', weight_sha256=WEIGHT_SHA256,
                   weight_enum=WEIGHT_ENUM, weight_url=WEIGHT_URL, seed=seed,
                   precision='float32', device='cpu', head_classes=5,
                   torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                   cpu_rng_before_sha256=before,
                   cpu_rng_after_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                   weight_source_reads=1, implicit_download=False,
                   limitation='Constructor identity only; no campaign/seed/ownership/budget gate.')
    return model, receipt
