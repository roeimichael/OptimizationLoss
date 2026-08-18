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


def update_experiment_status(experiment_path, status):
    config = load_config_from_path(experiment_path)
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


def get_experiments_by_status(results_dir='results'):
    by_status = {'pending': [], 'completed': []}
    for exp_path, config in get_all_experiment_configs(results_dir):
        key = 'completed' if config.get('status') == 'completed' else 'pending'
        by_status[key].append((exp_path, config))
    # rglob returns filesystem order, which is arbitrary and has in practice
    # come out grouped by ARM -- the one order that makes an interrupted
    # campaign unreadable. The generators all sort seed-major; restore it here.
    for key in by_status:
        by_status[key].sort(key=dispatch_key)
    return by_status


def print_status_summary(results_dir='results'):
    by_status = get_experiments_by_status(results_dir)
    total = len(by_status['completed']) + len(by_status['pending'])
    log.info("Experiments: %d total | %d completed | %d pending",
             total, len(by_status['completed']), len(by_status['pending']))
