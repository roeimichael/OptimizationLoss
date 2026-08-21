# Model caching: save/load warmup models to avoid redundant CE training.
# Cache key is base_model_id (hash of warmup-relevant hyperparameters).

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from src.models import get_model
from src.utils.error_handler import safe_execute

log = logging.getLogger(__name__)


def _amp_regime():
    """Which numeric regime this process would train under.

    Derived from the live device rather than from config['results'],
    because the warm-up is cached BEFORE any result is recorded.
    """
    try:
        import torch
        from src.pipeline.setup import runtime_provenance
        rt = runtime_provenance(torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))
        return "%s|scaler=%s" % (rt.get("amp_dtype"), rt.get("grad_scaler"))
    except Exception:
        return None


def get_cache_path(base_model_id: str) -> Path:
    """Repo-rooted, not CWD-relative: campaigns are launched from several
    working directories and a relative path silently gave each one its own
    cache (or, worse, made one dispatcher's cache invisible to the other)."""
    cache_dir = Path(os.environ.get("OPTLOSS_MODEL_CACHE")
                     or Path(__file__).resolve().parents[2] / "model_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{base_model_id}.pt"


def save_to_cache(model: nn.Module, base_model_id: str, config: Dict[str, Any]) -> None:
    """Write via a temp file + os.replace.

    Two dispatchers (one per GPU) share one NFS home and therefore one cache
    dir. Both can miss on the same id and both write it; a non-atomic torch.save
    lets the second one land on top of the first mid-write, and the loser reads
    a truncated file.
    """
    path = get_cache_path(base_model_id)
    payload = {
        'model_state_dict': model.state_dict(),
        'base_model_id': base_model_id,
        'code_version': config.get('code_version'),
        'data_fingerprint': config.get('data_fingerprint'),
        # The AMP regime is part of what trained these weights: the FP16
        # path SKIPS an overflowing optimizer step and BF16 does not, so
        # the same config takes a different number of steps on the two
        # servers. A warm-up shared across them is a silent regime mix, and
        # check_parity gate 4c cannot see it -- that gate reads each RUN's
        # recorded runtime, never the cache's.
        'amp_regime': _amp_regime(),
        'config': config,
        'saved_at': time.strftime('%Y-%m-%d'),
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    log.info("Model cached: %s", base_model_id)


def load_from_cache(base_model_id: str, config: Dict[str, Any],
                    num_classes: int, device: torch.device) -> Optional[nn.Module]:
    path = get_cache_path(base_model_id)
    if not path.exists():
        return None
    hp = config['hyperparams']
    ckpt = safe_execute(
        torch.load, path, map_location=device, weights_only=False,
        default=None, context=f"Loading cached model {base_model_id}"
    )
    if ckpt is None or ckpt.get('base_model_id') != base_model_id:
        return None
    # base_model_id hashes the hyperparameters, not the code. A cache written
    # before a change to what the warm-up OPTIMIZES is silently wrong -- exactly
    # how the pre-ImageNet-normalization caches survived a norm change.
    # The data behind a data_dir can change without the path changing.
    want_amp = _amp_regime()
    got_amp = ckpt.get('amp_regime')
    if want_amp and got_amp and got_amp != want_amp:
        log.warning("Cache %s was trained under AMP regime %s but this run "
                    "is %s -- the FP16 path skips overflowing optimizer "
                    "steps and BF16 does not, so they are not the same "
                    "warm-up. Retraining.", base_model_id, got_amp, want_amp)
        return None
    want_data = config.get('data_fingerprint')
    got_data = ckpt.get('data_fingerprint')
    if want_data and got_data != want_data:
        log.warning("Cache %s was trained on data fingerprint %s but this run "
                    "loaded %s -- the slice changed under the same path. "
                    "Retraining.", base_model_id,
                    got_data or "an unrecorded slice", want_data)
        return None
    want = config.get('code_version')
    got = ckpt.get('code_version')
    if want and got != want:
        log.warning("Cache %s was written by code_version %s but this run is "
                    "%s -- retraining rather than reusing it.",
                    base_model_id, got or "an unrecorded version", want)
        return None
    model = get_model(
        config['model_name'], n_classes=num_classes,
        dropout=hp['dropout'],
        pretrained=False
    ).to(device)
    result = safe_execute(
        model.load_state_dict, ckpt['model_state_dict'],
        default=None, context=f"Loading state dict for {base_model_id}"
    )
    return model if result is not None else None
