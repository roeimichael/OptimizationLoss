"""Exclusive claim records, not historical freshness or campaign permission.

The caller supplies an existing authenticated registry shared by its launchers.
Claims elsewhere, older operations and source/data/device/cost/budget gates are
outside this primitive. No scientific CLI imports or dispatches through it yet.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path


SCOPE = 'exclusive creation in this registry only; historical freshness, device, source and budget certification remain external'


def claim_seed(registry, seed, identity):
    """Create one kernel-exclusive name and never roll it back or overwrite it.

    Even an empty or partially written failed claim keeps the seed unavailable
    in this registry. A successful record is a declaration, not authentication
    of the supplied identity or authorization to execute a model/GPU campaign.
    """
    if type(seed) is not int or seed <= 0:
        raise ValueError('claim seed must be a positive integer')
    if not isinstance(identity, dict) or not identity or any(type(k) is not str for k in identity):
        raise ValueError('claim identity must be a nonempty JSON mapping')
    record = dict(format='exclusive_seed_claim_v1', seed=seed, declared_identity=identity,
                  created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope=SCOPE, campaign_permission=False)
    data = (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    path = Path(registry) / f'seed-{seed}.json'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    fd = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError('claim write made no progress')
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(file=str(path), sha256=hashlib.sha256(data).hexdigest())
