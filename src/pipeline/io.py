"""Shared IO: write config['results'] block + final status."""

import logging
import math

from src.utils.filesystem_manager import save_config_to_path

log = logging.getLogger(__name__)


def _non_finite(results):
    """Names of numeric results that are NaN or inf, recursively."""
    bad = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, "%s.%s" % (path, k) if path else str(k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, "%s[%d]" % (path, i))
        elif isinstance(obj, float) and not math.isfinite(obj):
            bad.append(path)

    walk(results, "")
    return bad


def save_results_to_config(config, experiment_path, results):
    """Write the results block and persist config.json.

    A diverged run is marked `diverged`, never `completed`. A NaN model still
    produces plausible-looking numbers -- argmax of all-NaN logits returns class
    0, which scores like a degenerate but healthy classifier -- and `completed`
    means the dispatcher never revisits it and the scorer counts it as coverage.
    That has happened: one all-NaN seed crashed the scorer and hid 23 healthy
    runs behind it.
    """
    config["results"] = results
    bad = _non_finite(results)
    if bad:
        config["status"] = "diverged"
        config["diverged_keys"] = bad
        log.error("DIVERGED: non-finite results %s -- writing status=diverged, "
                  "not completed", bad)
    else:
        config["status"] = "completed"
    save_config_to_path(config, experiment_path)
