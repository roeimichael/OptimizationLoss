"""Release-bound warm-up cache, with atomic publication and payload provenance."""
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile

import torch

from src.models import get_model
from src.pipeline.campaign import digest, release_runtime

log = logging.getLogger(__name__)


def _amp_regime():
    runtime = release_runtime()
    return '%s|scaler=%s' % (runtime['precision'], runtime['grad_scaler'])


def state_digest(state):
    """Canonical tensor names, shapes, dtypes and bytes, including buffers."""
    hashed = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        metadata = json.dumps([name, list(tensor.shape), str(tensor.dtype)],
                              separators=(',', ':')).encode()
        hashed.update(len(metadata).to_bytes(8, 'big'))
        hashed.update(metadata)
        hashed.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hashed.hexdigest()


def _identity(base_model_id, config):
    identity = config.get('cache_identity') or {}
    if not all(identity.get(k) for k in ('release_id', 'data_id')):
        return None
    if not config.get('data_fingerprint'):
        return None
    regime = _amp_regime()
    if not regime:
        return None
    return dict(identity, base_model_id=base_model_id,
                data_fingerprint=config['data_fingerprint'], amp_regime=regime)


def get_cache_path(base_model_id, config=None):
    """Fresh namespace only; missing identity can never load a historical cache."""
    identity = _identity(base_model_id, config or {})
    cache_dir = Path(os.environ.get('OPTLOSS_MODEL_CACHE') or
                     Path(__file__).resolve().parents[2]/'model_cache')/'fresh-v1'
    if identity is not None:
        cache_dir = cache_dir/digest(identity)
    return cache_dir/f'{base_model_id}.pt'


def _record(config, identity, artifact, state):
    config['warmup_checkpoint'] = dict(cache_identity=identity,
                                      artifact_sha256=artifact,
                                      state_sha256=state_digest(state))


def save_to_cache(model, base_model_id, config):
    identity = _identity(base_model_id, config)
    if identity is None:
        log.info('Warm-up not cached: complete fresh release/data/runtime identity required')
        return
    path = get_cache_path(base_model_id, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    payload = {'identity': identity, 'model_state_dict': state}
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w+b') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            artifact = hashlib.sha256(stream.read()).hexdigest()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    _record(config, identity, artifact, state)


def load_from_cache(base_model_id, config, num_classes, device):
    identity = _identity(base_model_id, config)
    if identity is None:
        return None
    path = get_cache_path(base_model_id, config)
    if not path.exists():
        return None
    # Hash the same opened payload passed to torch.load. A concurrent atomic
    # replacement cannot make the receipt describe other bytes.
    try:
        with path.open('rb') as stream:
            content = stream.read()
        checkpoint = torch.load(io.BytesIO(content), map_location=device, weights_only=False)
        if checkpoint.get('identity') != identity:
            return None
        model = get_model(config['model_name'], n_classes=num_classes,
                          dropout=config['hyperparams']['dropout'], pretrained=False).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except (OSError, RuntimeError, ValueError, KeyError, EOFError) as exc:
        log.warning('Cache refused: %s', exc)
        return None
    _record(config, identity, hashlib.sha256(content).hexdigest(), model.state_dict())
    return model
