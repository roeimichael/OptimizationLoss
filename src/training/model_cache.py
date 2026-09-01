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
        # The commit that RAN this warm-up, stamped by src/experiments/runner
        # at execution time. `code_version` is the generator's, written when
        # the config was created and never updated, so two runs either side of
        # a mid-campaign change to a training file agree on it and this cache
        # would be handed to both.
        'run_code_version': config.get('run_code_version'),
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
    # 🛑 BOTH SIDES ABSENT USED TO MEAN "NO CHECK", SILENTLY. This condition was
    # `want_amp and got_amp and got_amp != want_amp`, so it was and-chained on
    # its own inputs: `_amp_regime()` returning None (its except swallows any
    # failure) or a cache predating the field made the comparison not happen at
    # all, with no message. dsisco01 is FP16 + GradScaler and dsisco02 is BF16,
    # on a shared NFS home with ONE cache directory, so this is the check that
    # keeps a warm-up from crossing hosts. The third gate below (`run_code_
    # version`) already degrades explicitly and logs when it cannot run; these
    # two did not. Two of three.
    if not want_amp:
        # ⚠️ WARN, do not invalidate. `_amp_regime()` returns None whenever
        # the runtime probe fails, which is reachable off-GPU, and refusing the
        # cache there would retrain every warm-up on disk in exactly the
        # environment least able to tell whether that was necessary. The defect
        # being fixed is the SILENCE, not the reuse.
        log.warning("Cache %s: this process's AMP regime could not be "
                    "determined, so the FP16-vs-BF16 check DID NOT RUN. This "
                    "warm-up may have come from the other host.",
                    base_model_id)
    elif not got_amp:
        log.info("Cache %s predates `amp_regime`, so the FP16-vs-BF16 check "
                 "cannot run for it. Reusing it assumes one host.",
                 base_model_id)
    elif got_amp != want_amp:
        log.warning("Cache %s was trained under AMP regime %s but this run "
                    "is %s -- the FP16 path skips overflowing optimizer "
                    "steps and BF16 does not, so they are not the same "
                    "warm-up. Retraining.", base_model_id, got_amp, want_amp)
        return None
    want_data = config.get('data_fingerprint')
    got_data = ckpt.get('data_fingerprint')
    if not want_data:
        # Same shape: `want_data and ...` skipped the whole comparison when
        # this run had not recorded a fingerprint. Say so rather than passing.
        log.info("Cache %s: this run records no `data_fingerprint`, so the "
                 "slice-changed-under-the-same-path check DID NOT RUN.",
                 base_model_id)
    if want_data and got_data != want_data:
        log.warning("Cache %s was trained on data fingerprint %s but this run "
                    "loaded %s -- the slice changed under the same path. "
                    "Retraining.", base_model_id,
                    got_data or "an unrecorded slice", want_data)
        return None
    # PREFER THE RUNNER'S STAMP. It is the commit that produced these weights;
    # `code_version` is the commit that wrote the config, which is stamped once
    # at generation and never revisited, so it cannot see a change that landed
    # while the campaign was running.
    #
    # A cache written before the runner stamped anything carries no
    # `run_code_version`. That must NOT invalidate it -- every warm-up on disk
    # predates this field, and discarding them all would retrain the entire
    # cache. It degrades to the generator comparison, and says so.
    want_run = config.get('run_code_version')
    got_run = ckpt.get('run_code_version')
    if want_run and got_run:
        if got_run != want_run:
            log.warning("Cache %s was TRAINED by run_code_version %s but this "
                        "run is %s -- retraining rather than reusing it.",
                        base_model_id, got_run, want_run)
            return None
    else:
        want = config.get('code_version')
        got = ckpt.get('code_version')
        log.info("Cache %s carries no runner stamp (cache=%s, run=%s); falling "
                 "back to the GENERATOR's code_version, which cannot detect a "
                 "code change landed mid-campaign.",
                 base_model_id, got_run or "absent", want_run or "absent")
        if want and got != want:
            log.warning("Cache %s was written by code_version %s but this run "
                        "is %s -- retraining rather than reusing it.",
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
