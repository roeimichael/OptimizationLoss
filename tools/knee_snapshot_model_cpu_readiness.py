"""Actual ImageNet-cache constructor parity, CPU only and no data access.

Run in the isolated CPU namespace; this is no campaign or scientific seed claim.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
VALIDATION_SEED = 8080001


def main(weight_path, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from tralo.knee_snapshot_model import WEIGHT_SHA256, make_model
    path = Path(weight_path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != WEIGHT_SHA256:
        raise ValueError('actual pinned constructor fixture changed')
    import torch
    from torchvision import models
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('constructor CPU gate entered CUDA')
    # The reference is torchvision's native pretrained call, with its URL
    # provider replaced by these authenticated in-memory bytes. No network.
    original_provider = models.MobileNet_V3_Large_Weights.get_state_dict
    models.MobileNet_V3_Large_Weights.get_state_dict = lambda *a, **kw: torch.load(
        io.BytesIO(data), map_location='cpu', weights_only=True)
    try:
        torch.manual_seed(VALIDATION_SEED)
        reference = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        reference.classifier[3] = torch.nn.Linear(reference.classifier[3].in_features, 5)
        reference_rng = torch.get_rng_state().clone()
    finally:
        models.MobileNet_V3_Large_Weights.get_state_dict = original_provider
    actual, provenance = make_model(path, VALIDATION_SEED)
    native, produced = reference.state_dict(), actual.state_dict()
    if set(native) != set(produced) or any(not torch.equal(native[k], produced[k]) for k in native):
        raise RuntimeError('actual pretrained architecture/head initialization differs')
    if not torch.equal(reference_rng, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise RuntimeError('native RNG draw order or CPU boundary differs')
    def state_hash(values):
        digest = hashlib.sha256()
        for key in sorted(values):
            tensor = values[key].detach().contiguous().cpu()
            digest.update(key.encode()); digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()
    receipt = dict(status='passed',provenance=provenance,validation_seed=VALIDATION_SEED,
                   native_and_authenticated_state_sha256=state_hash(produced),state_entries=len(native),
                   exact_native_head_state_and_cpu_rng_equal=True,scientific_seed_claims=0,
                   actual_image_or_target_reads=0,cuda_initialized=torch.cuda.is_initialized(),gpu_hours=0,
                   elapsed_wall_seconds=time.monotonic()-started,
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   limitation='Real pretrained byte/constructor parity on CPU only. No forward/gradient/dose, real-input OS mount, campaign provenance logging, finite GPU cost or certified budget gate.')
    (output/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main(*sys.argv[1:])
