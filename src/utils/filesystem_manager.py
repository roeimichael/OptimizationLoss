"""Experiment filesystem management utilities."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any


def save_config_to_path(config: Dict[str, Any], experiment_path: str) -> str:
    """Save config to experiment directory."""
    config_path = Path(experiment_path) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return str(config_path)


def load_config_from_path(experiment_path: str) -> Dict[str, Any]:
    """Load config from experiment directory."""
    config_path = Path(experiment_path) / 'config.json'
    with open(config_path, 'r') as f:
        return json.load(f)


def get_all_experiment_configs(results_dir: str = 'results') -> List[Tuple[str, Dict]]:
    """Get all experiment configs from results directory."""
    experiments = []
    results_path = Path(results_dir)

    if not results_path.exists():
        return experiments

    for config_file in results_path.rglob('config.json'):
        try:
            config = load_config_from_path(config_file.parent)
            experiments.append((str(config_file.parent), config))
        except Exception as e:
            print(f"Warning: Failed to load {config_file}: {e}")

    return experiments


def update_experiment_status(experiment_path: str, status: str) -> None:
    """Update experiment status in config."""
    config = load_config_from_path(experiment_path)
    config['status'] = status
    save_config_to_path(config, experiment_path)


def save_stop_reason(experiment_path: str, status: str, reason: str,
                     final_epoch: int = None, global_satisfied: bool = None,
                     local_satisfied: bool = None) -> None:
    """Save stop reason and final state to config."""
    config = load_config_from_path(experiment_path)

    config['run_completion'] = {
        'status': status,
        'reason': reason,
        'final_epoch': final_epoch,
        'global_constraint_satisfied': global_satisfied,
        'local_constraint_satisfied': local_satisfied,
        'completed_at': datetime.now().isoformat()
    }

    config['status'] = 'completed' if status == 'converged' else 'pending'
    save_config_to_path(config, experiment_path)


def get_experiments_by_status(results_dir: str = 'results') -> Dict[str, List]:
    """Group experiments by status."""
    by_status = {'pending': [], 'completed': []}

    for exp_path, config in get_all_experiment_configs(results_dir):
        status = config.get('status', 'pending')
        key = 'completed' if status == 'completed' else 'pending'
        by_status[key].append((exp_path, config))

    return by_status


def print_status_summary(results_dir: str = 'results') -> None:
    """Print experiment status summary."""
    by_status = get_experiments_by_status(results_dir)
    total = len(by_status['completed']) + len(by_status['pending'])

    print(f"Experiments: {total} total | {len(by_status['completed'])} completed | {len(by_status['pending'])} pending")
    print("=" * 60)
