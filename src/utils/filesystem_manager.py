# Experiment filesystem management: config save/load and status tracking.
# Supports scanning experiment directories for pending/completed experiments.

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def save_config_to_path(config, experiment_path):
    config_path = Path(experiment_path) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return str(config_path)


def load_config_from_path(experiment_path):
    with open(Path(experiment_path) / 'config.json', 'r') as f:
        return json.load(f)


def get_all_experiment_configs(results_dir='results'):
    experiments = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return experiments
    for config_file in results_path.rglob('config.json'):
        try:
            config = load_config_from_path(config_file.parent)
            experiments.append((str(config_file.parent), config))
        except Exception as e:
            log.warning("Failed to load %s: %s", config_file, e)
    return experiments


def update_experiment_status(experiment_path, status, count_failure=False):
    """Write the status back to config.json.

    `count_failure` increments a persistent counter. A config that fails
    deterministically -- a bad hyperparameter, a missing file -- used to reset
    to `pending` and be picked up again by every subsequent dispatch, forever,
    with nothing on disk saying why. After MAX_FAILURES it is marked `failed`
    and skipped until a human resets it.
    """
    config = load_config_from_path(experiment_path)
    if count_failure:
        config['failures'] = int(config.get('failures', 0)) + 1
        if config['failures'] >= MAX_FAILURES:
            status = 'failed'
            log.error("%s has now failed %d times -- marking `failed` so it "
                      "stops being re-dispatched. Fix it and reset the status "
                      "to `pending` to retry.", experiment_path, config['failures'])
    config['status'] = status
    save_config_to_path(config, experiment_path)




def dispatch_key(item):
    """Seed-major ordering key for a (path, config) pair.

    Seed first, so every arm finishes seed 1 before any arm starts seed 2 and
    an interrupted campaign leaves matched slices rather than one complete arm
    with an empty control. Cross-campaign drift here is ~0.027, twice the size
    of the effects at issue, so arms are only readable against arms from the
    same campaign -- a finished arm whose control never ran is not a partial
    result, it is none.

    Everything after the seed is a stable tie-break, and the path is last so
    the order is total even for configs missing these keys.
    """
    _path, cfg = item
    hp = cfg.get('hyperparams') or {}
    return (
        hp.get('seed') if isinstance(hp.get('seed'), int) else 1 << 30,
        str(cfg.get('model_name', '')),
        str(cfg.get('dataset_mode', '')),
        str(cfg.get('constraint_tag', '')),
        str(cfg.get('arm', '')),
        str(_path),
    )


MAX_FAILURES = 3
# Statuses that must NOT be re-dispatched. `running` is deliberately absent:
# an interrupted run resets to pending, which is what makes overnight
# re-dispatch idempotent. These two are different -- retrying them cannot help.
TERMINAL = {'completed', 'failed', 'diverged'}


def get_experiments_by_status(results_dir='results'):
    by_status = {'pending': [], 'completed': [], 'blocked': []}
    for exp_path, config in get_all_experiment_configs(results_dir):
        status = config.get('status')
        if status == 'completed':
            key = 'completed'
        elif status in TERMINAL or config.get('failures', 0) >= MAX_FAILURES:
            key = 'blocked'
        else:
            key = 'pending'
        by_status[key].append((exp_path, config))
    # rglob returns filesystem order, which is arbitrary and has in practice
    # come out grouped by ARM -- the one order that makes an interrupted
    # campaign unreadable. The generators all sort seed-major; restore it here.
    for key in by_status:
        by_status[key].sort(key=dispatch_key)
    return by_status


def print_status_summary(results_dir='results'):
    by_status = get_experiments_by_status(results_dir)
    blocked = by_status.get('blocked', [])
    total = len(by_status['completed']) + len(by_status['pending']) + len(blocked)
    log.info("Experiments: %d total | %d completed | %d pending | %d blocked",
             total, len(by_status['completed']), len(by_status['pending']),
             len(blocked))
    if blocked:
        log.warning("%d run(s) will NOT be re-dispatched (failed or diverged). "
                    "See error_log.json in each; reset status to `pending` to "
                    "retry:", len(blocked))
        for path, cfg in blocked[:5]:
            log.warning("   %s  status=%s failures=%s",
                        path, cfg.get('status'), cfg.get('failures', 0))
