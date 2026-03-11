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


def get_experiments_by_status(results_dir='results'):
    by_status = {'pending': [], 'completed': []}
    for exp_path, config in get_all_experiment_configs(results_dir):
        key = 'completed' if config.get('status') == 'completed' else 'pending'
        by_status[key].append((exp_path, config))
    return by_status


def print_status_summary(results_dir='results'):
    by_status = get_experiments_by_status(results_dir)
    total = len(by_status['completed']) + len(by_status['pending'])
    log.info("Experiments: %d total | %d completed | %d pending",
             total, len(by_status['completed']), len(by_status['pending']))
